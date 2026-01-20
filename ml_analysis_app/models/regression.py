from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from config.settings import Config

class RegressionModels:
    @staticmethod
    def run_linear_regression(X, y, test_size=Config.TEST_SIZE_DEFAULT):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=Config.RANDOM_STATE
        )
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'coefficient': lr.coef_[0],
            'intercept': lr.intercept_,
            'mse': mse,
            'r2': r2,
            'model': lr,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    @staticmethod
    def run_multiple_regression(X, y, test_size=Config.TEST_SIZE_DEFAULT):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=Config.RANDOM_STATE
        )
        
        mr = LinearRegression()
        mr.fit(X_train, y_train)
        y_pred = mr.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'coefficients': dict(zip(X.columns, mr.coef_)),
            'intercept': mr.intercept_,
            'mse': mse,
            'r2': r2
        }