from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from config.settings import Config

class ClassificationModels:
    @staticmethod
    def run_knn(X, y, test_size=Config.TEST_SIZE_DEFAULT):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=Config.RANDOM_STATE
        )
        
        best_k = 1
        best_accuracy = 0
        k_accuracies = {}
        
        for k in range(1, Config.KNN_MAX_K + 1):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            k_accuracies[k] = acc
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_k = k
        
        return {
            'best_k': best_k,
            'accuracy': best_accuracy,
            'k_accuracies': k_accuracies
        }
    
    @staticmethod
    def run_naive_bayes(X, y, test_size=Config.TEST_SIZE_DEFAULT):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=Config.RANDOM_STATE
        )
        
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {'accuracy': accuracy}
    
    @staticmethod
    def run_decision_tree(X, y, test_size=Config.TEST_SIZE_DEFAULT):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=Config.RANDOM_STATE
        )
        
        dt = DecisionTreeClassifier(criterion='entropy', random_state=Config.RANDOM_STATE)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'feature_importance': dict(zip(X.columns, dt.feature_importances_))
        }
    
    @staticmethod
    def run_neural_network(X, y, test_size=Config.TEST_SIZE_DEFAULT):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=Config.RANDOM_STATE
        )
        
        nn = MLPClassifier(
            hidden_layer_sizes=Config.NN_HIDDEN_LAYERS,
            max_iter=Config.NN_MAX_ITER,
            random_state=Config.RANDOM_STATE
        )
        nn.fit(X_train, y_train)
        y_pred = nn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'iterations': nn.n_iter_
        }