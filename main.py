from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import xgboost as xgb
import pandas as pd
import pickle
import joblib

# Initialize the FastAPI app
app = FastAPI()

# Load the trained XGB model
model = joblib.load('xgb_model.pkl')

# Define the input data models
class SelectedVariant(BaseModel):
    name: Optional[str]
    type: Optional[str]
    value: Optional[str]

class CartItem(BaseModel):
    product: Dict[str, Any]
    quantity: int
    selectedVariants: List[SelectedVariant]
    pricePerUnit: float

class OrderTotal(BaseModel):
    deliveryCost: Optional[float] = 0
    deliveryPrice: Optional[float] = 0
    totalPrice: float

class Customer(BaseModel):
    phone: str
    address: Optional[str] = None
    name: Optional[str] = None
    ip: Optional[str] = None
    city: Optional[str] = None
    userAgent: Optional[str] = None

class HistoryItem(BaseModel):
    status: Optional[str] = None
    timestamp: Optional[str] = None
    actionTaker: Optional[str] = None
    rejectionReason: Optional[str] = None

class Store(BaseModel):
    _id: str
    name: Optional[str] = None
    slug: Optional[str] = None
    domain: Optional[str] = None

class Order(BaseModel):
    _id: str
    reference: int
    customer: Customer
    status: Optional[str] = "pending"
    attempt: Optional[int] = None
    note: Optional[str] = None
    cart: Optional[List[CartItem]] = None
    total: Optional[OrderTotal] = None
    deliveryCompany: Optional[str] = None
    barcode: Optional[str] = None
    label: Optional[Dict[str, Any]] = None
    history: Optional[List[HistoryItem]] = None
    store: Optional[Union[str, Store]] = None
    archived: Optional[bool] = False
    duplicated: Optional[bool] = False
    expiryDate: Optional[str] = None
    analyticsData: Optional[Dict[str, Any]] = None
    paymentStatus: Optional[str] = None
    billedAmount: Optional[float] = None
    refunded: Optional[bool] = False
    isTest: Optional[bool] = False
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None

# Preprocessing functions
def aggregate_duplicates(cart_df, duplicates):
    result = {}
    for product_id in duplicates:
        subset = cart_df[cart_df['product'].map(lambda x: x.get('_id') or x.get('$oid')) == product_id]
        total_quantity = subset['quantity'].sum()
        price_variation = subset['pricePerUnit'].std() if len(subset) > 1 else 0
        result[product_id] = {
            'total_quantity': total_quantity,
            'price_variation': price_variation
        }
    return result

def add_duplicate_features(order):
    if order['cart'] is None:
        order['cart'] = []
    cart_df = pd.DataFrame(order['cart'])
    if cart_df.empty:
        order['num_duplicate_products'] = 0
        order['total_quantity_duplicates'] = 0
        order['avg_price_variation_duplicates'] = 0
        return order
    duplicate_counts = cart_df['product'].map(lambda x: x.get('_id') or x.get('$oid')).value_counts()
    duplicates = duplicate_counts[duplicate_counts > 1].index.tolist()
    aggregated_duplicates = aggregate_duplicates(cart_df, duplicates)

    # Add features to order data
    order['num_duplicate_products'] = len(duplicates)
    order['total_quantity_duplicates'] = sum([agg['total_quantity'] for agg in aggregated_duplicates.values()])
    order['avg_price_variation_duplicates'] = sum([agg['price_variation'] for agg in aggregated_duplicates.values()]) / len(duplicates) if len(duplicates) > 0 else 0

    return order

def add_valid_phone_number_feature(orders):
    allowed_prefixes = ['2', '3', '4', '5', '6', '7', '9']

    def is_valid_phone_number(phone_number):
        if phone_number is None or phone_number == '':
            return False
        if phone_number[0] in allowed_prefixes and (len(phone_number) == 8 or len(phone_number) == 9):
            return True
        return False

    orders['phone_number_valid'] = orders['customer_phone'].apply(is_valid_phone_number)

def add_updated_at_is_greater_than_5_minutes(orders):
    # Ensure both times are timezone-naive
    now = pd.Timestamp.now().tz_localize(None)
    orders['updated_at_is_greater_than_5_minutes'] = (now - orders['updatedAt'].dt.tz_localize(None)) > pd.Timedelta(minutes=5)

def optimize_data_types(data):
    data["status"] = data["status"].astype("category")
    data["deliveryCompany"] = data["deliveryCompany"].astype("category")
    data["paymentStatus"] = data["paymentStatus"].astype("category")

    data["billedAmount"] = data["billedAmount"].astype("float32")
    data["total_price"] = data["total_price"].astype("float32")
    data["avg_price_variation_duplicates"] = data["avg_price_variation_duplicates"].astype("float32")

    data["total_quantity"] = data["total_quantity"].astype("int32")
    data["status_refund"] = data["status_refund"].astype("int8")
    data["num_duplicate_products"] = data["num_duplicate_products"].astype("int8")
    data["total_quantity_duplicates"] = data["total_quantity_duplicates"].astype("int8")
    data["attempt"] = data["attempt"].astype("int8")

    data["refunded"] = data["refunded"].astype("bool")

# Prediction endpoint
@app.post("/predict")
def predict(orders: List[Order]):
    try:
        # Convert list of orders to DataFrame
        orders_df = pd.DataFrame([order.dict() for order in orders])

        # Handle missing values
        orders_df['archived'] = orders_df['archived'].fillna(False)
        orders_df['duplicated'] = orders_df['duplicated'].fillna(False)
        orders_df['status_refund'] = 0
        orders_df['attempt'] = orders_df['attempt'].fillna(0)

        # Preprocess the data
        orders_df['createdAt'] = pd.to_datetime(orders_df['createdAt'])
        orders_df['updatedAt'] = pd.to_datetime(orders_df['updatedAt'])
        orders_df['cart'] = orders_df['cart'].apply(lambda x: x if x is not None else [])
        orders_df['total_quantity'] = orders_df['cart'].apply(lambda cart: sum(item['quantity'] for item in cart))
        orders_df['customer_phone'] = orders_df['customer'].apply(lambda x: x.get('phone', ''))
        orders_df['total_price'] = orders_df['total'].apply(lambda x: x.get('totalPrice', 0.0))
        orders_df = orders_df.apply(add_duplicate_features, axis=1)
        add_valid_phone_number_feature(orders_df)
        add_updated_at_is_greater_than_5_minutes(orders_df)
        optimize_data_types(orders_df)

        # Define the features to keep for prediction
        features = ['attempt', 'duplicated', 'status_refund', 'total_price', 'total_quantity', 'billedAmount', 'avg_price_variation_duplicates', 
                    'num_duplicate_products', 'total_quantity_duplicates', 'phone_number_valid',
                    'status', 'deliveryCompany', 'paymentStatus', 'updated_at_is_greater_than_5_minutes']

        orders_df = orders_df[features]
        
        predictions = model.predict(orders_df)
        
        # make predictions list and round the values
        predictions = [round(value) for value in predictions.tolist()]
        
        # Response with predictions
        return {"predictions": predictions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
