"""
Sample Data Generator

This module creates sample datasets for testing the data preparation pipeline.
It generates realistic data with various quality issues to demonstrate the pipeline's capabilities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class SampleDataGenerator:
    """
    Generate sample datasets with various data quality issues for testing
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the sample data generator
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_customer_data(self, num_records: int = 1000) -> pd.DataFrame:
        """
        Generate sample customer data with various quality issues
        
        Args:
            num_records: Number of customer records to generate
            
        Returns:
            pd.DataFrame: Sample customer dataset
        """
        logger.info(f"Generating {num_records} customer records")
        
        # Generate base customer data
        customers = []
        
        for i in range(num_records):
            # Generate customer ID
            customer_id = f"CUST_{i+1:05d}"
            
            # Generate names (with some inconsistencies)
            first_names = ['John', 'Jane', 'Mike', 'Sarah', 'David', 'Lisa', 'Tom', 'Emma', 'Chris', 'Anna']
            last_names = ['Smith', 'Johnson', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor', 'Anderson', 'Thomas']
            
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            
            # Add some case inconsistencies
            if random.random() < 0.1:
                first_name = first_name.lower()
            if random.random() < 0.1:
                last_name = last_name.upper()
            
            # Add some whitespace issues
            if random.random() < 0.05:
                first_name = f" {first_name} "
            if random.random() < 0.05:
                last_name = f" {last_name}"
            
            # Generate email (with some invalid formats)
            if random.random() < 0.95:
                email = f"{first_name.strip().lower()}.{last_name.strip().lower()}@{'gmail.com' if random.random() < 0.6 else 'yahoo.com'}"
            else:
                # Invalid email format
                email = f"{first_name.strip().lower()}.{last_name.strip().lower()}@invalid"
            
            # Generate phone numbers (with inconsistent formats)
            area_codes = ['123', '456', '789', '555', '999']
            area_code = random.choice(area_codes)
            phone_formats = [
                f"({area_code}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
                f"{area_code}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                f"{area_code}.{random.randint(100, 999)}.{random.randint(1000, 9999)}",
                f"{area_code}{random.randint(100, 999)}{random.randint(1000, 9999)}"
            ]
            phone = random.choice(phone_formats)
            
            # Generate age (with some outliers)
            if random.random() < 0.95:
                age = random.randint(18, 80)
            else:
                # Outlier ages
                age = random.choice([5, 150, 200])
            
            # Generate income (with some outliers)
            if random.random() < 0.9:
                income = random.randint(25000, 150000)
            else:
                # Outlier incomes
                income = random.choice([5000, 500000, 1000000])
            
            # Generate registration date
            start_date = datetime(2020, 1, 1)
            end_date = datetime(2023, 12, 31)
            random_date = start_date + timedelta(
                days=random.randint(0, (end_date - start_date).days)
            )
            
            # Generate account status
            status_options = ['Active', 'Inactive', 'Suspended', 'Pending']
            status = random.choice(status_options)
            
            # Generate city (with some inconsistencies)
            cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                     'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
            city = random.choice(cities)
            
            # Add some inconsistencies
            if random.random() < 0.1:
                city = city.lower()
            if random.random() < 0.05:
                city = f" {city} "
            
            # Generate state
            states = ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'TX', 'CA', 'TX', 'CA']
            state = random.choice(states)
            
            # Generate purchase amount (with some missing values)
            if random.random() < 0.85:
                purchase_amount = round(random.uniform(10, 2000), 2)
            else:
                purchase_amount = None
            
            # Generate loyalty score (stored as string sometimes)
            loyalty_score = random.randint(1, 100)
            if random.random() < 0.2:
                loyalty_score = str(loyalty_score)
            
            customer = {
                'customer_id': customer_id,
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'phone': phone,
                'age': age,
                'income': income,
                'registration_date': random_date.strftime('%Y-%m-%d'),
                'status': status,
                'city': city,
                'state': state,
                'purchase_amount': purchase_amount,
                'loyalty_score': loyalty_score
            }
            
            customers.append(customer)
        
        # Create DataFrame
        df = pd.DataFrame(customers)
        
        # Add some duplicate records
        num_duplicates = int(num_records * 0.05)  # 5% duplicates
        duplicate_indices = np.random.choice(df.index, num_duplicates, replace=False)
        duplicates = df.iloc[duplicate_indices].copy()
        df = pd.concat([df, duplicates], ignore_index=True)
        
        # Add some completely missing rows
        num_missing_rows = int(num_records * 0.02)  # 2% missing rows
        for _ in range(num_missing_rows):
            missing_row = {col: None for col in df.columns}
            missing_row['customer_id'] = f"CUST_{len(df)+1:05d}"
            df = pd.concat([df, pd.DataFrame([missing_row])], ignore_index=True)
        
        logger.info(f"Generated customer dataset with shape: {df.shape}")
        return df
    
    def generate_sales_data(self, num_records: int = 2000) -> pd.DataFrame:
        """
        Generate sample sales data with various quality issues
        
        Args:
            num_records: Number of sales records to generate
            
        Returns:
            pd.DataFrame: Sample sales dataset
        """
        logger.info(f"Generating {num_records} sales records")
        
        sales = []
        
        for i in range(num_records):
            # Generate transaction ID
            transaction_id = f"TXN_{i+1:06d}"
            
            # Generate customer ID (some may not exist)
            customer_id = f"CUST_{random.randint(1, 1200):05d}"
            
            # Generate product information
            products = [
                'Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Keyboard',
                'Mouse', 'Monitor', 'Webcam', 'Printer', 'Router'
            ]
            categories = [
                'Electronics', 'Computers', 'Accessories', 'Networking', 'Peripherals'
            ]
            
            product_name = random.choice(products)
            category = random.choice(categories)
            
            # Generate quantity (with some unrealistic values)
            if random.random() < 0.9:
                quantity = random.randint(1, 10)
            else:
                # Unrealistic quantities
                quantity = random.choice([0, -1, 1000])
            
            # Generate price (with some outliers)
            if random.random() < 0.9:
                price = round(random.uniform(10, 2000), 2)
            else:
                # Outlier prices
                price = random.choice([0, -100, 50000])
            
            # Calculate total amount
            total_amount = quantity * price
            
            # Generate sales date
            start_date = datetime(2020, 1, 1)
            end_date = datetime(2023, 12, 31)
            sale_date = start_date + timedelta(
                days=random.randint(0, (end_date - start_date).days)
            )
            
            # Generate salesperson (with inconsistent formats)
            salesperson_names = ['Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Brown', 'Eve Wilson']
            salesperson = random.choice(salesperson_names)
            
            # Add some inconsistencies
            if random.random() < 0.1:
                salesperson = salesperson.lower()
            if random.random() < 0.05:
                salesperson = f" {salesperson} "
            
            # Generate region
            regions = ['North', 'South', 'East', 'West', 'Central']
            region = random.choice(regions)
            
            # Generate discount percentage (with missing values)
            if random.random() < 0.7:
                discount_percent = round(random.uniform(0, 30), 1)
            else:
                discount_percent = None
            
            # Generate payment method
            payment_methods = ['Credit Card', 'Debit Card', 'Cash', 'PayPal', 'Bank Transfer']
            payment_method = random.choice(payment_methods)
            
            # Generate status
            status_options = ['Completed', 'Pending', 'Cancelled', 'Refunded']
            status = random.choice(status_options)
            
            sale = {
                'transaction_id': transaction_id,
                'customer_id': customer_id,
                'product_name': product_name,
                'category': category,
                'quantity': quantity,
                'price': price,
                'total_amount': total_amount,
                'sale_date': sale_date.strftime('%Y-%m-%d'),
                'salesperson': salesperson,
                'region': region,
                'discount_percent': discount_percent,
                'payment_method': payment_method,
                'status': status
            }
            
            sales.append(sale)
        
        # Create DataFrame
        df = pd.DataFrame(sales)
        
        # Add some duplicate records
        num_duplicates = int(num_records * 0.03)  # 3% duplicates
        duplicate_indices = np.random.choice(df.index, num_duplicates, replace=False)
        duplicates = df.iloc[duplicate_indices].copy()
        df = pd.concat([df, duplicates], ignore_index=True)
        
        logger.info(f"Generated sales dataset with shape: {df.shape}")
        return df
    
    def generate_product_data(self, num_records: int = 500) -> pd.DataFrame:
        """
        Generate sample product data with various quality issues
        
        Args:
            num_records: Number of product records to generate
            
        Returns:
            pd.DataFrame: Sample product dataset
        """
        logger.info(f"Generating {num_records} product records")
        
        products = []
        
        for i in range(num_records):
            # Generate product ID
            product_id = f"PROD_{i+1:04d}"
            
            # Generate product name
            product_types = ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Keyboard',
                           'Mouse', 'Monitor', 'Webcam', 'Printer', 'Router']
            brands = ['Apple', 'Samsung', 'HP', 'Dell', 'Sony', 'Microsoft', 'Logitech']
            
            product_type = random.choice(product_types)
            brand = random.choice(brands)
            model = f"{brand} {product_type} {random.randint(100, 999)}"
            
            # Generate category
            categories = ['Electronics', 'Computers', 'Accessories', 'Networking', 'Peripherals']
            category = random.choice(categories)
            
            # Generate price (with some missing values and outliers)
            if random.random() < 0.9:
                if random.random() < 0.95:
                    price = round(random.uniform(10, 2000), 2)
                else:
                    # Outlier prices
                    price = random.choice([0, -50, 50000])
            else:
                price = None
            
            # Generate cost (should be less than price)
            if price is not None and price > 0:
                cost = round(price * random.uniform(0.3, 0.8), 2)
            else:
                cost = None
            
            # Generate stock quantity (with some negative values)
            if random.random() < 0.95:
                stock_quantity = random.randint(0, 1000)
            else:
                # Negative stock (data error)
                stock_quantity = random.randint(-100, -1)
            
            # Generate weight (with some missing values)
            if random.random() < 0.8:
                weight = round(random.uniform(0.1, 10.0), 2)
            else:
                weight = None
            
            # Generate dimensions (inconsistent units)
            if random.random() < 0.7:
                length = round(random.uniform(5, 50), 1)
                width = round(random.uniform(5, 50), 1)
                height = round(random.uniform(1, 20), 1)
                
                # Sometimes use different units
                if random.random() < 0.2:
                    dimensions = f"{length}x{width}x{height} inches"
                else:
                    dimensions = f"{length}x{width}x{height} cm"
            else:
                dimensions = None
            
            # Generate rating (with some invalid values)
            if random.random() < 0.8:
                if random.random() < 0.95:
                    rating = round(random.uniform(1, 5), 1)
                else:
                    # Invalid ratings
                    rating = random.choice([0, 6, 10])
            else:
                rating = None
            
            # Generate review count
            if rating is not None:
                review_count = random.randint(0, 1000)
            else:
                review_count = None
            
            # Generate supplier (with inconsistent formats)
            suppliers = ['ABC Electronics', 'XYZ Tech', 'Global Supply Co', 'Tech Distributors Inc']
            supplier = random.choice(suppliers)
            
            # Add some inconsistencies
            if random.random() < 0.1:
                supplier = supplier.lower()
            if random.random() < 0.05:
                supplier = f" {supplier} "
            
            # Generate availability status
            availability_options = ['In Stock', 'Out of Stock', 'Discontinued', 'Pre-order']
            availability = random.choice(availability_options)
            
            # Generate creation date
            start_date = datetime(2018, 1, 1)
            end_date = datetime(2023, 12, 31)
            created_date = start_date + timedelta(
                days=random.randint(0, (end_date - start_date).days)
            )
            
            product = {
                'product_id': product_id,
                'product_name': model,
                'category': category,
                'price': price,
                'cost': cost,
                'stock_quantity': stock_quantity,
                'weight': weight,
                'dimensions': dimensions,
                'rating': rating,
                'review_count': review_count,
                'supplier': supplier,
                'availability': availability,
                'created_date': created_date.strftime('%Y-%m-%d')
            }
            
            products.append(product)
        
        # Create DataFrame
        df = pd.DataFrame(products)
        
        # Add some duplicate records
        num_duplicates = int(num_records * 0.04)  # 4% duplicates
        duplicate_indices = np.random.choice(df.index, num_duplicates, replace=False)
        duplicates = df.iloc[duplicate_indices].copy()
        df = pd.concat([df, duplicates], ignore_index=True)
        
        logger.info(f"Generated product dataset with shape: {df.shape}")
        return df
    
    def save_sample_datasets(self, output_dir: str = "data") -> Dict[str, str]:
        """
        Generate and save all sample datasets
        
        Args:
            output_dir: Directory to save the datasets
            
        Returns:
            Dict mapping dataset names to file paths
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        datasets = {
            'customers': self.generate_customer_data(1000),
            'sales': self.generate_sales_data(2000),
            'products': self.generate_product_data(500)
        }
        
        file_paths = {}
        
        for name, dataset in datasets.items():
            file_path = os.path.join(output_dir, f"{name}.csv")
            dataset.to_csv(file_path, index=False)
            file_paths[name] = file_path
            logger.info(f"Saved {name} dataset to {file_path}")
        
        return file_paths


def main():
    """Main function to generate sample datasets"""
    generator = SampleDataGenerator()
    file_paths = generator.save_sample_datasets()
    
    print("Sample datasets generated successfully:")
    for name, path in file_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()