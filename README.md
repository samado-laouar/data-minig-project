# 🤖 Machine Learning Data Analysis Platform

A comprehensive, user-friendly web application for data analysis and machine learning built with Streamlit. This platform provides a step-by-step workflow for data preprocessing and running various ML algorithms with beautiful visualizations.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-FF4B4B.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-F7931E.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

### 📁 **Step 1: Data Import**

- Support for **CSV** and **Excel** files
- Interactive data preview with tabs
- Comprehensive dataset statistics
- Column type detection and missing value analysis

### 🔧 **Step 2: Data Preprocessing**

- **Automatic Mode**: One-click preprocessing with standard techniques
- **Manual Mode**: Full control over data cleaning
  - Missing value handling (mean, median, mode, custom fill)
  - Outlier detection and removal (IQR method)
  - Data normalization (StandardScaler)
  - Duplicate removal

### 🚀 **Step 3: Machine Learning Algorithms**

#### Classification Algorithms:

- **K-Nearest Neighbors (KNN)**: Automatic K optimization (1-20)
- **Naive Bayes**: Gaussian Naive Bayes classifier
- **Decision Tree C4.5**: With feature importance visualization
- **Neural Network**: Multi-layer perceptron classifier

#### Regression Algorithms:

- **Linear Regression**: Two-variable analysis with visualization
- **Multiple Regression**: All features with coefficient analysis

### 📊 **Step 4: Results & Analysis**

- Performance metrics for all algorithms
- Interactive comparison charts
- Detailed algorithm breakdowns
- Downloadable results report (CSV)

## 🏗️ Project Structure

```
ml_analysis_app/
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── config/
│   └── settings.py            # Configuration and constants
├── utils/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading utilities
│   ├── preprocessor.py        # Preprocessing functions
│   └── visualization.py       # Plotly visualization functions
├── models/
│   ├── __init__.py
│   ├── classification.py      # Classification algorithms
│   └── regression.py          # Regression algorithms
└── pages/
    ├── __init__.py
    ├── data_import.py         # Step 1: Data import page
    ├── preprocessing.py       # Step 2: Preprocessing page
    ├── algorithms.py          # Step 3: Algorithms page
    └── results.py             # Step 4: Results page
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/ml-analysis-app.git
   cd ml-analysis-app
   ```

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

## 📖 Usage Guide

### Step-by-Step Workflow

1. **Upload Your Dataset**

   - Click on "Browse files" or drag and drop your CSV/Excel file
   - Preview your data and check statistics
   - Click **Next** to proceed

2. **Preprocess Your Data**

   - Choose **Automatic** for quick preprocessing
   - Choose **Manual** for custom preprocessing options
   - Review preprocessed data
   - Click **Next** to proceed

3. **Run Algorithms**

   - Select an algorithm from the available options
   - Choose target variable
   - Adjust parameters (test size, etc.)
   - Run algorithm and view results
   - Repeat for multiple algorithms
   - Click **Next** when done

4. **View Results**
   - Review performance metrics
   - Compare algorithms
   - Download results report

### Navigation

- **Next ➡️**: Proceed to next step (enabled when current step is complete)
- **⬅️ Back**: Return to previous step
- **🔄 Start Over**: Reset the entire workflow (sidebar)

## 🎨 Screenshots

### Data Import

![Data Import](screenshots/data_import.png)

### Preprocessing

![Preprocessing](screenshots/preprocessing.png)

### Algorithms

![Algorithms](screenshots/algorithms.png)

### Results

![Results](screenshots/results.png)

## 📊 Supported Algorithms

| Algorithm           | Type           | Key Features                                 |
| ------------------- | -------------- | -------------------------------------------- |
| KNN                 | Classification | Auto K-optimization (1-20), Accuracy metrics |
| Naive Bayes         | Classification | Fast probabilistic classifier                |
| Decision Tree C4.5  | Classification | Feature importance, Entropy-based splits     |
| Neural Network      | Classification | MLP with configurable layers                 |
| Linear Regression   | Regression     | Two-variable analysis, Visual plots          |
| Multiple Regression | Regression     | Multi-feature analysis, Coefficient ranking  |

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/)**: Web application framework
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation and analysis
- **[NumPy](https://numpy.org/)**: Numerical computing
- **[Scikit-learn](https://scikit-learn.org/)**: Machine learning algorithms
- **[Plotly](https://plotly.com/)**: Interactive visualizations
- **[OpenPyXL](https://openpyxl.readthedocs.io/)**: Excel file support

## 📋 Requirements

```
streamlit==1.31.0
pandas==2.1.4
numpy==1.26.3
scikit-learn==1.4.0
plotly==5.18.0
openpyxl==3.1.2
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - _Initial work_ - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- Scikit-learn contributors for ML algorithms
- Plotly for beautiful visualizations

## 📧 Contact

For questions or suggestions, please open an issue or contact [your.email@example.com](mailto:your.email@example.com)

## 🔮 Future Enhancements

- [ ] More ML algorithms (SVM, Random Forest, XGBoost)
- [ ] Advanced feature engineering tools
- [ ] Model persistence (save/load models)
- [ ] Automated hyperparameter tuning
- [ ] Cross-validation support
- [ ] Data visualization dashboard
- [ ] Export to Jupyter Notebook
- [ ] Multi-language support

---

⭐ **Star this repo** if you find it helpful!

**Made with ❤️ using Streamlit**
