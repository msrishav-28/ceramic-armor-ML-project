from setuptools import setup, find_packages

setup(
    name="ceramic_armor_ml",
    version="1.0.0",
    description="Machine Learning Pipeline for Ceramic Armor Property Prediction",
    author="M S Rishav Subhin",
    author_email="msrishav28@gmail.com",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.26.4",
        "pandas>=2.2.0",
        "scipy>=1.12.0",
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.3",
        "catboost>=1.2.3",
        "optuna>=3.5.0",
        "shap>=0.44.1",
        "matplotlib>=3.8.2",
        "seaborn>=0.13.2",
        "plotly>=5.18.0",
        "pymatgen>=2024.2.8",
        "mp-api>=0.41.2",
        "joblib>=1.3.2",
        "pyyaml>=6.0.1",
        "loguru>=0.7.2",
        "tqdm>=4.66.1",
        "scikit-learn-intelex>=2024.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "black>=24.1.1",
            "flake8>=7.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Materials Science",
        "Programming Language :: Python :: 3.11",
    ],
)
