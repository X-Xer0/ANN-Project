"""
Stock Price Prediction & Movement Classification using Artificial Neural Networks
================================================================================
Author: AI Assistant
Python Version: 3.10+
Framework: TensorFlow/Keras

This project combines regression (price prediction) and classification (movement prediction)
using ANNs on historical stock data with technical indicators.
"""

import os
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

# Data processing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                           accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')

# Technical Analysis
import ta

# =======================================================================================
# 1. DATA COLLECTION MODULE
# =======================================================================================

class StockDataCollector:
    """Handles stock data collection from Yahoo Finance"""
    
    def __init__(self, ticker, start_date=None, end_date=None):
        self.ticker = ticker.upper()
        self.end_date = end_date or datetime.now()
        self.start_date = start_date or (self.end_date - timedelta(days=365*10))
        
    def fetch_data(self, save_path='stock_data.csv'):
        """Fetch historical stock data from Yahoo Finance"""
        print(f"\n{'='*60}")
        print(f"Fetching data for {self.ticker}...")
        print(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        print('='*60)
        
        try:
            # Download data
            stock_data = yf.download(
                self.ticker,
                start=self.start_date,
                end=self.end_date,
                progress=True
            )
            
            if stock_data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
            
            # Reset index to have Date as column
            stock_data.reset_index(inplace=True)
            
            # Save to CSV
            stock_data.to_csv(save_path, index=False)
            print(f"✓ Data saved to {save_path}")
            print(f"✓ Shape: {stock_data.shape}")
            print(f"✓ Date range: {stock_data['Date'].min()} to {stock_data['Date'].max()}")
            
            return stock_data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

# =======================================================================================
# 2. FEATURE ENGINEERING MODULE
# =======================================================================================

class FeatureEngineer:
    """Creates technical indicators and features for the model"""
    
    def __init__(self, df):
        self.df = df.copy()
    
    def __init__(self, df):
        self.df = df.copy()

        # If yfinance returned MultiIndex columns (Price, Ticker), flatten to single level
        if isinstance(self.df.columns, pd.MultiIndex):
            # keep only the top level ("Price"), which yields columns like 'Date','Close','Open',...
            self.df.columns = self.df.columns.get_level_values(0)
        
    def add_technical_indicators(self):
        """Add various technical indicators to the dataset"""
        print("\n📊 Adding Technical Indicators...")
        
        # Moving Averages
        self.df['MA_10'] = self.df['Close'].rolling(window=10).mean()
        self.df['MA_20'] = self.df['Close'].rolling(window=20).mean()
        self.df['MA_50'] = self.df['Close'].rolling(window=50).mean()
        
        # Price ratios to moving averages
        self.df['Price_MA10_Ratio'] = self.df['Close'] / self.df['MA_10']
        self.df['Price_MA20_Ratio'] = self.df['Close'] / self.df['MA_20']
        self.df['Price_MA50_Ratio'] = self.df['Close'] / self.df['MA_50']
        
        # RSI (Relative Strength Index)
        self.df['RSI'] = ta.momentum.RSIIndicator(self.df['Close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(self.df['Close'])
        self.df['MACD'] = macd.macd()
        self.df['MACD_Signal'] = macd.macd_signal()
        self.df['MACD_Diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(self.df['Close'], window=20, window_dev=2)
        self.df['BB_Upper'] = bb.bollinger_hband()
        self.df['BB_Lower'] = bb.bollinger_lband()
        self.df['BB_Middle'] = bb.bollinger_mavg()
        self.df['BB_Width'] = self.df['BB_Upper'] - self.df['BB_Lower']
        self.df['BB_Position'] = (self.df['Close'] - self.df['BB_Lower']) / (self.df['BB_Upper'] - self.df['BB_Lower'])
        
        # Volume indicators
        self.df['Volume_MA'] = self.df['Volume'].rolling(window=10).mean()
        self.df['Volume_Ratio'] = self.df['Volume'] / self.df['Volume_MA']
        
        # Price features
        self.df['High_Low_Pct'] = (self.df['High'] - self.df['Low']) / self.df['Close'] * 100
        self.df['Close_Open_Pct'] = (self.df['Close'] - self.df['Open']) / self.df['Open'] * 100
        
        # Volatility
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Volatility'] = self.df['Returns'].rolling(window=20).std()
        
        # ATR (Average True Range)
        self.df['ATR'] = ta.volatility.AverageTrueRange(
            self.df['High'], self.df['Low'], self.df['Close'], window=14
        ).average_true_range()
        
        # Drop NaN values
        self.df.dropna(inplace=True)
        
        print(f"✓ Added {len(self.df.columns) - 6} technical indicators")
        print(f"✓ Final dataset shape: {self.df.shape}")
        
        return self.df
    
    def create_labels(self):
        """Create labels for classification (UP/DOWN movement)"""
        # Next day's close price (for regression)
        self.df['Next_Close'] = self.df['Close'].shift(-1)
        
        # Movement classification (1: UP, 0: DOWN)
        self.df['Movement'] = (self.df['Next_Close'] > self.df['Close']).astype(int)
        
        # Remove last row (no next day)
        self.df = self.df[:-1]
        
        print(f"✓ Created labels: Next_Close (regression), Movement (classification)")
        print(f"✓ Class distribution: UP={self.df['Movement'].sum()}, DOWN={len(self.df)-self.df['Movement'].sum()}")
        
        return self.df

# =======================================================================================
# 3. DATA PREPARATION MODULE
# =======================================================================================

class DataPreprocessor:
    """Handles data preprocessing and preparation for neural networks"""
    
    def __init__(self, df, sequence_length=60):
        self.df = df.copy()
        self.sequence_length = sequence_length
        self.scalers = {}
        
    def select_features(self):
        """Select relevant features for modeling"""
        # Features to use
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_10', 'MA_20', 'MA_50',
            'Price_MA10_Ratio', 'Price_MA20_Ratio', 'Price_MA50_Ratio',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff',
            'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
            'Volume_Ratio', 'High_Low_Pct', 'Close_Open_Pct',
            'Returns', 'Volatility', 'ATR'
        ]
        
        # Target columns
        target_regression = 'Next_Close'
        target_classification = 'Movement'
        
        return feature_cols, target_regression, target_classification
    
    def scale_data(self, X, y_reg, feature_name='features'):
        """Scale features and targets using MinMaxScaler"""
        # Scale features
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler_X.fit_transform(X)
        self.scalers[f'{feature_name}_X'] = scaler_X
        
        # Scale regression target
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        y_reg_scaled = scaler_y.fit_transform(y_reg.reshape(-1, 1))
        self.scalers[f'{feature_name}_y'] = scaler_y
        
        return X_scaled, y_reg_scaled.flatten()
    
    def create_sequences(self, X, y_reg, y_class):
        """Create sequences for time series prediction"""
        X_seq, y_reg_seq, y_class_seq = [], [], []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_reg_seq.append(y_reg[i])
            y_class_seq.append(y_class[i])
        
        return np.array(X_seq), np.array(y_reg_seq), np.array(y_class_seq)
    
    def prepare_data(self, test_size=0.15, val_size=0.15):
        """Prepare data for training, validation, and testing"""
        print("\n🔧 Preparing Data...")
        
        # Select features
        feature_cols, target_reg, target_class = self.select_features()
        
        # Extract features and targets
        X = self.df[feature_cols].values
        y_reg = self.df[target_reg].values
        y_class = self.df[target_class].values
        
        # Scale data
        X_scaled, y_reg_scaled = self.scale_data(X, y_reg)
        
        # Create sequences
        X_seq, y_reg_seq, y_class_seq = self.create_sequences(X_scaled, y_reg_scaled, y_class)
        
        # Split data: 70% train, 15% val, 15% test
        # First split: train+val vs test
        test_split = int(len(X_seq) * (1 - test_size))
        X_temp, X_test = X_seq[:test_split], X_seq[test_split:]
        y_reg_temp, y_reg_test = y_reg_seq[:test_split], y_reg_seq[test_split:]
        y_class_temp, y_class_test = y_class_seq[:test_split], y_class_seq[test_split:]
        
        # Second split: train vs val
        val_split = int(len(X_temp) * (1 - val_size/(1-test_size)))
        X_train, X_val = X_temp[:val_split], X_temp[val_split:]
        y_reg_train, y_reg_val = y_reg_temp[:val_split], y_reg_temp[val_split:]
        y_class_train, y_class_val = y_class_temp[:val_split], y_class_temp[val_split:]
        
        print(f"✓ Train set: {X_train.shape}")
        print(f"✓ Validation set: {X_val.shape}")
        print(f"✓ Test set: {X_test.shape}")
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_reg_train': y_reg_train, 'y_reg_val': y_reg_val, 'y_reg_test': y_reg_test,
            'y_class_train': y_class_train, 'y_class_val': y_class_val, 'y_class_test': y_class_test
        }

# =======================================================================================
# 4. MODEL BUILDING MODULE
# =======================================================================================

class StockPredictionModels:
    """Build and compile ANN models for regression and classification"""
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    def build_regression_model(self):
        """Build ANN for stock price prediction (regression)"""
        model = models.Sequential([
            # Input layer
            layers.LSTM(128, return_sequences=True, input_shape=self.input_shape),
            layers.Dropout(0.2),
            
            # Hidden layers
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            
            # Output layer
            layers.Dense(1, activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        print("\n🏗️ Regression Model Architecture:")
        model.summary()
        
        return model
    
    def build_classification_model(self):
        """Build ANN for movement classification"""
        model = models.Sequential([
            # Input layer
            layers.LSTM(128, return_sequences=True, input_shape=self.input_shape),
            layers.Dropout(0.3),
            
            # Hidden layers
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        print("\n🏗️ Classification Model Architecture:")
        model.summary()
        
        return model

# =======================================================================================
# 5. MODEL TRAINING MODULE
# =======================================================================================

class ModelTrainer:
    """Handle model training with callbacks"""
    
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.history = None
        
    def get_callbacks(self):
        """Define training callbacks"""
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f'{self.model_name}_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model"""
        print(f"\n🚀 Training {self.model_name}...")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title(f'{self.model_name} - Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Metric plot (MAE for regression, Accuracy for classification)
        if 'accuracy' in self.history.history:
            axes[1].plot(self.history.history['accuracy'], label='Training Accuracy')
            axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            axes[1].set_ylabel('Accuracy')
        else:
            axes[1].plot(self.history.history['mae'], label='Training MAE')
            axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
            axes[1].set_ylabel('MAE')
        
        axes[1].set_title(f'{self.model_name} - Performance')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.model_name}_training_history.png', dpi=100)
        plt.show()

# =======================================================================================
# 6. MODEL EVALUATION MODULE
# =======================================================================================

class ModelEvaluator:
    """Evaluate model performance"""
    
    def __init__(self, model, model_type='regression'):
        self.model = model
        self.model_type = model_type
        
    def evaluate_regression(self, X_test, y_test, scaler_y):
        """Evaluate regression model"""
        print("\n📈 Regression Model Evaluation:")
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_test)
        
        # Inverse transform predictions
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
        mae = mean_absolute_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        
        print(f"✓ RMSE: ${rmse:.2f}")
        print(f"✓ MAE: ${mae:.2f}")
        print(f"✓ R² Score: {r2:.4f}")
        
        # Plot predictions
        self.plot_regression_results(y_test_original, y_pred)
        
        return {
            'predictions': y_pred,
            'actual': y_test_original,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def evaluate_classification(self, X_test, y_test):
        """Evaluate classification model"""
        print("\n📊 Classification Model Evaluation:")
        
        # Make predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"✓ Accuracy: {accuracy:.4f}")
        print(f"✓ Precision: {precision:.4f}")
        print(f"✓ Recall: {recall:.4f}")
        print(f"✓ F1 Score: {f1:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        return {
            'predictions': y_pred,
            'probabilities': y_pred_prob,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def plot_regression_results(self, y_true, y_pred, num_points=100):
        """Plot regression results"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Actual vs Predicted
        axes[0].scatter(y_true[:num_points], y_pred[:num_points], alpha=0.6)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Price')
        axes[0].set_ylabel('Predicted Price')
        axes[0].set_title('Actual vs Predicted Stock Prices')
        axes[0].grid(True)
        
        # Time series plot
        axes[1].plot(y_true[:num_points], label='Actual', alpha=0.7)
        axes[1].plot(y_pred[:num_points], label='Predicted', alpha=0.7)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Stock Price')
        axes[1].set_title('Stock Price Prediction Over Time')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('regression_results.png', dpi=100)
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['DOWN', 'UP'], 
                    yticklabels=['DOWN', 'UP'])
        plt.title('Confusion Matrix - Movement Classification')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix.png', dpi=100)
        plt.show()

# =======================================================================================
# 7. VISUALIZATION MODULE
# =======================================================================================

class StockVisualizer:
    """Advanced visualization for stock analysis"""
    
    def __init__(self, df, ticker):
        self.df = df
        self.ticker = ticker
        
    def plot_stock_overview(self):
        """Create comprehensive stock analysis visualization"""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Price and Volume
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(self.df['Date'], self.df['Close'], label='Close Price', linewidth=2)
        ax1.fill_between(self.df['Date'], self.df['Low'], self.df['High'], alpha=0.3)
        ax1.set_title(f'{self.ticker} - Stock Price', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Moving Averages
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(self.df['Date'], self.df['Close'], label='Close', alpha=0.7)
        ax2.plot(self.df['Date'], self.df['MA_10'], label='MA10', alpha=0.7)
        ax2.plot(self.df['Date'], self.df['MA_20'], label='MA20', alpha=0.7)
        ax2.plot(self.df['Date'], self.df['MA_50'], label='MA50', alpha=0.7)
        ax2.set_title('Moving Averages', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. RSI
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(self.df['Date'], self.df['RSI'], label='RSI', color='orange', linewidth=2)
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax3.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax3.fill_between(self.df['Date'], 30, 70, alpha=0.1)
        ax3.set_title('Relative Strength Index (RSI)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. MACD
        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(self.df['Date'], self.df['MACD'], label='MACD', linewidth=2)
        ax4.plot(self.df['Date'], self.df['MACD_Signal'], label='Signal', linewidth=2)
        ax4.bar(self.df['Date'], self.df['MACD_Diff'], label='Histogram', alpha=0.3)
        ax4.set_title('MACD Indicator', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('MACD')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Bollinger Bands
        ax5 = plt.subplot(3, 2, 5)
        ax5.plot(self.df['Date'], self.df['Close'], label='Close', color='black', linewidth=1)
        ax5.plot(self.df['Date'], self.df['BB_Upper'], label='Upper Band', alpha=0.7)
        ax5.plot(self.df['Date'], self.df['BB_Lower'], label='Lower Band', alpha=0.7)
        ax5.fill_between(self.df['Date'], self.df['BB_Lower'], self.df['BB_Upper'], alpha=0.2)
        ax5.set_title('Bollinger Bands', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Price ($)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Volume
        ax6 = plt.subplot(3, 2, 6)
        ax6.bar(self.df['Date'], self.df['Volume'], alpha=0.7, width=1)
        ax6.plot(self.df['Date'], self.df['Volume_MA'], color='red', label='Volume MA', linewidth=2)
        ax6.set_title('Trading Volume', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Volume')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.ticker} - Technical Analysis Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.ticker}_technical_analysis.png', dpi=100)
        plt.show()
        
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap of features"""
        # Select numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numerical_cols].corr()
        
        plt.figure(figsize=(20, 16))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=100)
        plt.show()

# =======================================================================================
# 8. MAIN PIPELINE
# =======================================================================================

def main():
    """Main execution pipeline"""
    
    print("\n" + "="*60)
    print("🤖 STOCK PRICE PREDICTION & MOVEMENT CLASSIFICATION")
    print("    Using Artificial Neural Networks (ANNs)")
    print("="*60)
    
    # Get user input
    ticker = input("\n📊 Enter stock ticker (e.g., AAPL, TSLA, MSFT): ").strip().upper()
    if not ticker:
        ticker = "AAPL"
        print(f"Using default ticker: {ticker}")
    
    # =================== STEP 1: DATA COLLECTION ===================
    collector = StockDataCollector(ticker)
    df = collector.fetch_data()
    
    if df is None:
        print("Failed to fetch data. Exiting...")
        return
    
    # =================== STEP 2: FEATURE ENGINEERING ===================
    engineer = FeatureEngineer(df)
    df = engineer.add_technical_indicators()
    df = engineer.create_labels()
    
    # =================== STEP 3: DATA PREPARATION ===================
    preprocessor = DataPreprocessor(df, sequence_length=60)
    data = preprocessor.prepare_data()
    
    # =================== STEP 4: BUILD MODELS ===================
    input_shape = (data['X_train'].shape[1], data['X_train'].shape[2])
    
    model_builder = StockPredictionModels(input_shape)
    regression_model = model_builder.build_regression_model()
    classification_model = model_builder.build_classification_model()
    
    # =================== STEP 5: TRAIN MODELS ===================
    # Train Regression Model
    regression_trainer = ModelTrainer(regression_model, 'Price_Prediction_Model')
    regression_history = regression_trainer.train(
        data['X_train'], data['y_reg_train'],
        data['X_val'], data['y_reg_val'],
        epochs=50,
        batch_size=32
    )
    regression_trainer.plot_training_history()
    
    # Train Classification Model
    classification_trainer = ModelTrainer(classification_model, 'Movement_Classification_Model')
    classification_history = classification_trainer.train(
        data['X_train'], data['y_class_train'],
        data['X_val'], data['y_class_val'],
        epochs=50,
        batch_size=32
    )
    classification_trainer.plot_training_history()
    
    # =================== STEP 6: EVALUATE MODELS ===================
    # Evaluate Regression Model
    regression_evaluator = ModelEvaluator(regression_model, 'regression')
    regression_results = regression_evaluator.evaluate_regression(
        data['X_test'], 
        data['y_reg_test'],
        preprocessor.scalers['features_y']
    )
    
    # Evaluate Classification Model
    classification_evaluator = ModelEvaluator(classification_model, 'classification')
    classification_results = classification_evaluator.evaluate_classification(
        data['X_test'],
        data['y_class_test']
    )
    
    # =================== STEP 7: VISUALIZATIONS ===================
    visualizer = StockVisualizer(df, ticker)
    visualizer.plot_stock_overview()
    visualizer.plot_correlation_heatmap()
    
    # =================== STEP 8: SAVE MODELS ===================
    regression_model.save(f'{ticker}_price_prediction_model.h5')
    classification_model.save(f'{ticker}_movement_classification_model.h5')
    print(f"\n✓ Models saved successfully!")
    
    # =================== STEP 9: FINAL SUMMARY ===================
    print("\n" + "="*60)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"\n🎯 Stock Ticker: {ticker}")
    print(f"📅 Data Range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"📈 Total Trading Days: {len(df)}")
    
    print("\n🤖 REGRESSION MODEL (Price Prediction):")
    print(f"   • RMSE: ${regression_results['rmse']:.2f}")
    print(f"   • MAE: ${regression_results['mae']:.2f}")
    print(f"   • R² Score: {regression_results['r2']:.4f}")
    
    print("\n🎯 CLASSIFICATION MODEL (Movement Prediction):")
    print(f"   • Accuracy: {classification_results['accuracy']:.4f}")
    print(f"   • Precision: {classification_results['precision']:.4f}")
    print(f"   • Recall: {classification_results['recall']:.4f}")
    print(f"   • F1 Score: {classification_results['f1']:.4f}")
    
    print("\n✅ Project completed successfully!")
    print("📁 Generated files:")
    print("   • stock_data.csv")
    print("   • {ticker}_price_prediction_model.h5")
    print("   • {ticker}_movement_classification_model.h5")
    print("   • Various visualization plots (.png)")
    
    return {
        'ticker': ticker,
        'data': df,
        'models': {
            'regression': regression_model,
            'classification': classification_model
        },
        'results': {
            'regression': regression_results,
            'classification': classification_results
        }
    }

# =======================================================================================
# 9. EXECUTION
# =======================================================================================

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run main pipeline
    results = main()
    
    print("\n🚀 To deploy this model, check the Streamlit app: streamlit_app.py")
    print("💡 Run: streamlit run streamlit_app.py")
