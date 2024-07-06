import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


# =======================================================PRE-PROCESSING===============================================================
class PREPROCESS:
    def __init__(self, data_list):
        self.data_list = data_list
        self.df = None

        
    def load_and_df(self) :
        df_list = []
        for dataset_index in range(len(self.data_list)) :
            data = open(self.data_list[dataset_index]).readlines()
            df = pd.DataFrame(data)
            df_list.append(df)
        self.df = pd.concat(df_list, ignore_index=True)
        # list of all variables
        self.list_column = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak',
                            'ST_Slope', 'CountMajorVessels', 'Thalassemia', 'HeartDisease', 'Area']
        for column_index in  range(len(self.list_column)): 
            self.df[self.list_column[column_index]]  = self.df.iloc[:,0].apply(lambda x: x.strip().split('/')[column_index])
            
        self.df.drop(0,axis='columns',inplace = True)
        self.df.reset_index(drop=True, inplace=True)
        
        print('===================Unique Values of the columns===================')
        for column_index in  self.list_column: 
            print('--------------This is:',column_index,'--------------')
            list_valuess = self.df[column_index].unique().tolist()
            list_valuess.sort()
            print(list_valuess)
            print('-----------------------------------')
        return self.df, self.list_column
        
    def convert_whitespace_zero(self) :
        # replacing the whitespaces with np.nan
        print('===================Replacing whitespaces with np.nan===================')
        self.df = self.df.replace(' ',np.nan)
        
        print('===================Replacing 0 in the Possitive Float Columns(Age, RestingBP, Cholesterol, MaxHR) with np.nan===================')
        # list of continuous postitive variables
        list_possitive_floats = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
        print('-------------------Checking the Narutal Number Columns-------------------')
        list_possitive_float_zero = []
        for col in list_possitive_floats: 
            zero_counts = self.df[col].isin(['0','0.0']).sum()
            if zero_counts > 0 :  
                print(f"{col}+'has {zero_counts} zero values")
                list_possitive_float_zero.append(col)
                print('Replacing the zeroes with np.nan')
                self.df[col] = self.df[col].replace('0',np.nan)
                print('---------------DONE--------------')
        return self.df

    def convert_dtype(self) :
        print('===================Encoding and Coverting dtype for categorcial columns so that they are nullable===================')
        # list of categorical variables
        self.list_categorical = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 
                                 'ST_Slope', 'CountMajorVessels', 'Thalassemia', 'HeartDisease', 'Area']
        for column in self.list_categorical:
            print('---------------This is:',column,'---------------')
            self.df[column] = pd.Categorical(self.df[column])
            print(dict(enumerate(self.df[column].cat.categories)))
            self.df[column] = self.df[column].cat.codes
        # default cat.codes converts the values to Int8, so I convert them to Int32 so that all the integer in the sample has a cosistent type of Int32
            self.df[column] = self.df[column].astype('Int32')
        # cat.codes converts NaN to -1, but for now I want to keep them as np.na.
            self.df[column].replace(-1, np.nan, inplace=True)
        print('===================Converting dtypes of continuous columns so that they are nullable===================')
        
        #Convert dtype of 'Oldpeak' to Float32
        # list of continuous variables
        self.list_continuous = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        print('---------------Oldpeak: Float32---------------')
        self.df[self.list_continuous] = self.df[self.list_continuous].astype('Float32')
        print(self.df.info())
        
        return self.df, self.list_categorical, self.list_continuous










# ========================================================EDA================================================================================
class EDA:
    def __init__(self, df):
        self.df = df
        self.list_column = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak',
                            'ST_Slope', 'CountMajorVessels', 'Thalassemia', 'HeartDisease', 'Area']
        # list of categorical variables
        self.list_categorical = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 
                                 'ST_Slope', 'CountMajorVessels', 'Thalassemia', 'HeartDisease', 'Area']
        # list of continuous variables
        self.list_continuous = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    
    def missing_check(self):
        print('The number of missing values in each column:')
        print(self.df.isna().sum())
        for col in self.list_column:
            percentage_missing = round(self.df[col].isna().sum() / self.df.shape[0] * 100, 2)
            print('Percentage of missing values in ' + col + ': ' + str(percentage_missing) + '%')
        # List of variables having missing values
        list_missing = [col for col in  self.list_column if self.df[col].isnull().sum() > 0]
        # List of categorical variables having missing values
        list_categorical_missing = [col for col in  list_missing if col in self.list_categorical]
        # List of continuous variables having missing values
        list_continuous_missing = [col for col in  list_missing if col in self.list_continuous]
    
        from scipy.stats import chi2_contingency
    
        # Detect missing values and create a binary indicator
        missing_indicators = self.df.isnull().astype(int)
        print('===========================================================================================================')
        print('Perform Chi-Squared hypothesis testing to determine the mechanism of the missing data in the categorical variables')
        print('H0 (Null Hypothesis): The missingness of a categorical variable (var1) is independent of the missingness of another categorical variable (var2) (indicating MCAR if true across all variables.')
    
        print('Alternative Hypothesis (H1): The missingness of a categorical variable (var1) is not independent of the missingness of another categorical variable (var2) (indicating MAR or MNAR).')
    
        print('alpha = 0.05')
        results = []
        for i in range(len(self.list_categorical)):
            for j in range(i + 1, len(self.list_categorical)):
                var1 = self.list_categorical[i]
                var2 = self.list_categorical[j]
    
                # Create a contingency table
                contingency_table = pd.crosstab(missing_indicators[var1], missing_indicators[var2])
    
                # Perform the chi-squared test
                chi2, p, dof, ex = chi2_contingency(contingency_table, correction=False)
    
                # Append the result
                results.append({
                    'Variable 1': var1,
                    'Variable 2': var2,
                    'Chi-Squared': chi2,
                    'p-value': p,
                    'Degrees of Freedom': dof
                })
    
        # Print full DataFrame horizontally
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
    
        # Convert results to DataFrame for easy viewing
        results_df = pd.DataFrame(results)
    
        # Display results
        print(results_df)
    
        # Interpret the results and print only those with no significant relationship
        alpha = 0.05
        print("\nCases where we reject the null hypothesis (significant relationship between missingness):\n")
        list_categorical_missing_depen = []
        for index, row in results_df.iterrows():
            if row['p-value'] < alpha:
                if row['Variable 1'] not in list_categorical_missing_depen:
                    list_categorical_missing_depen.append(row['Variable 1'])
                if row['Variable 2'] not in list_categorical_missing_depen:
                    list_categorical_missing_depen.append(row['Variable 2'])
                print(f"Variable 1: {row['Variable 1']}, Variable 2: {row['Variable 2']}, Chi-Squared: {row['Chi-Squared']:.4f}, p-value: {row['p-value']:.4f}, Degrees of Freedom: {row['Degrees of Freedom']}")
        return list_missing, list_categorical_missing_depen, list_continuous_missing
    
    
    def check_ANCOVA_linearity(self, input_df) :
        sns.set(style="ticks")  
        fig,axs = plt.subplots(10,10, figsize = (50,50))
        df_plot = input_df.dropna().reindex()
        n=0
        m=0
        list_continuous_covariate = self.list_continuous.copy()
        for i in range(len(self.list_continuous)) :
                list_continuous_covariate.remove(self.list_continuous[i])
                for col in list_continuous_covariate :
                    for cat in self.list_categorical :
                        ax = sns.scatterplot(x=col, y= self.list_continuous[i] , 
                                            hue= cat, data=df_plot,
                                            ax=axs[n, m]
                                            )
                        m+=1
                        if m==10 :
                            m=0
                            n+=1
                        ax.get_legend().set_visible(False)
        fig.tight_layout()    
    
    
    def missing_impute_df(self, input_df, list_missing, list_categorical_missing_depen) :
        for col in list_categorical_missing_depen:
            if col in self.list_categorical:
                print(f"{col} has dependent missingness")
        print('==========================================================================================================')
        print('Impute the missing values of CATEGORICAL variables whose missingness is NOT independent of each other using LinearRegression')
        # return list_dependent_missingness_variables
        df_cat_depen = input_df[list_categorical_missing_depen]
        for target_col in list_categorical_missing_depen :
            # Separate the data into training (non-missing target) and testing (missing target) sets
            train_data = df_cat_depen[df_cat_depen[target_col].notnull()]
            test_data = df_cat_depen[df_cat_depen[target_col].isnull()]
    
            # Define input features and target for training
            X_train = train_data.drop(columns=[target_col])
            y_train = train_data[target_col]
    
            # Define input features for testing
            X_test = test_data.drop(columns=[target_col])
    
            # Impute missing values in the input features using the median
            imputer = SimpleImputer(strategy='median')
            X_train_imputed = imputer.fit_transform(X_train)
    
            # Check if there are missing values in X_test
            if X_test.isnull().any().any():
                # Handle missing values in X_test appropriately
                X_test_imputed = imputer.transform(X_test)
            else:
                X_test_imputed = X_test  # No missing values, use as is
    
            # Train the model
            model = LinearRegression()
            model.fit(X_train_imputed, y_train)
    
            # Predict the missing values and update the dataframe
            predicted_values = model.predict(X_test_imputed)
    
            # Convert predicted values to integers if the target column is of integer type. 
            if df_cat_depen[target_col].dtype == pd.Int32Dtype():
                predicted_values = np.round(predicted_values)
                predicted_values = pd.Series(predicted_values, dtype=pd.Int32Dtype())
                predicted_values.index = df_cat_depen[df_cat_depen[target_col].isna()].index
            df_cat_depen.loc[df_cat_depen[target_col].isnull(), target_col] = predicted_values
    
        
        # Impute missing values of Integer or Continuous Columns using Random Forest 
        print('==========================================================================================================')
        print('Impute the missing values of CONTINUOUS variables and CATEGORICAL variables whose missingness is INDEPENDENT of other Categorcial variables using Random Forest ')
        from sklearn.ensemble import RandomForestRegressor
        # the variables to be imputed with Random Forest are in the list missing_columns but not those in list_categorical_missing_depen (imouted with Linear Regression)
        list_rf_imputed_var = [col for col in list_missing if col not in list_categorical_missing_depen]
        df_rf = self.df.copy()

        # Create an imputer for the features
        feature_imputer = SimpleImputer(strategy='median')

        for col in list_rf_imputed_var:
            # Prepare the data
            # Features: All variables except the target variable
            X = df_rf.drop(columns=[col])
    
            # Target variable
            y = df_rf[col]
    
            # Split the data into two subsets: complete observations and missing values
            complete_data_mask = ~y.isnull()
            X_complete = X[complete_data_mask]
            y_complete = y[complete_data_mask]
            X_missing = X[~complete_data_mask]

            # Impute missing values in the features
            X_complete_imputed = feature_imputer.fit_transform(X_complete)
            X_missing_imputed = feature_imputer.transform(X_missing)
    
            # Train a Random Forest model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_complete_imputed, y_complete)
    
            # Predict missing values
            y_missing_pred = rf_model.predict(X_missing_imputed)
    
            # Convert predicted values to integers if the target column is of integer type. 
            if df_rf[col].dtype == pd.Int32Dtype():
                y_missing_pred = np.round(y_missing_pred)
                y_missing_pred = pd.Series(y_missing_pred, dtype=pd.Int32Dtype())
            
            # Re-index the y_missing_pred so that they match the indexes of the missing value of col in df
                y_missing_pred.index = df_rf[df_rf[col].isna()].index
                indexing = y_missing_pred.index.to_list()

            # Impute missing values
            df_rf.loc[~complete_data_mask, col] = y_missing_pred
        
        input_df[list_categorical_missing_depen] = df_cat_depen[list_categorical_missing_depen]
        input_df[list_rf_imputed_var] = df_rf[list_rf_imputed_var]
        print('Done')
        return input_df
    

    
    def distribution_con(self,input_df) :
        sns.set(style="ticks")
        fig, axs = plt.subplots(3, 2, figsize=(20, 20))
        n = 0
        m = 0
        for column in self.list_continuous:
            # Loop through the non-numeric columns.
            # Draw histogram of the column
            df_plot = self.df[input_df[column].notnull()]
            ax = sns.histplot(data=df_plot,
                              x=df_plot[column],
                              # bins = len(df_plot[column].unique().tolist()),
                              # binrange=(df_plot[column].min(), df[df[column].notnull()][column].max()+1),
                              color='#4285F4',
                              stat='density',
                              element='step',
                              ax=axs[n, m]
                              )
            # A vertical line for the median of the column
            col_median = input_df[column].median()
            ax.axvline(x=col_median,
                       color='tab:red',
                       linestyle=':',
                       label='median of sample',
                       )
            ax.legend(loc='upper right')
            m += 1
            if m == 2:
                m = 0
                n += 1
            # plt.title(str(column))
            # plt.show()
        fig.tight_layout()
        from IPython.display import display, clear_output
        display(fig)
        plt.close(fig)

    def outliers_check(self, input_df) :
        sns.set(style="ticks")  
        fig,axs = plt.subplots(3,2, figsize = (20,20))
        n=0
        m=0
        for column in self.list_continuous : 
            ax = sns.boxplot(data = input_df,
                            x= input_df[column],
                            color = '#4285F4',
                            ax=axs[n, m]
                            )
            m+=1
            if m==2 :
                m=0
                n+=1
            # plt.title(str(column))
            # plt.show()
        fig.tight_layout()
        from IPython.display import display, clear_output
        display(fig)
        plt.close(fig)

    def outliers_address(self, input_df) :
        # Define percentile thresholds for Winsorization
        lower_percentile = 0.05
        upper_percentile = 0.95
        df_winsor = input_df.copy()
        print('==================================Apply Winsorization to each numeric column:=========================')
        # Apply Winsorization to each numeric column
        for col in self.list_continuous:
            lower_limit = df_winsor[col].quantile(lower_percentile)
            upper_limit = df_winsor[col].quantile(upper_percentile)
            df_winsor[col] = np.where(df_winsor[col] < lower_limit, lower_limit, 
                                      np.where(df_winsor[col] > upper_limit, upper_limit, df_winsor[col])
                                      )
            # Display summary statistics after Winsorization
        summary_stats = df_winsor[self.list_continuous].describe()
        print(summary_stats)
        print('-----------------------------------Recheck Outliers with Boxplots:------------------------------------')
        self.outliers_check(df_winsor)
        
        print('-----------------------------------Recheck the Distribution with Histograms:-----------------------')
        self.distribution_con(df_winsor)


        print('=================================Apply median imputation to each numeric column:==========================')
        df_med = input_df.copy()
        for col in self.list_continuous:
            # Calculate the median of each column
            median = df_med[col].median()
            
            # Calculate the IQR (Interquartile Range) for each column
            q1 = df_med[col].quantile(0.25)
            q3 = df_med[col].quantile(0.75)
            iqr = q3 - q1
            
            # Define lower and upper bounds for outlier detection
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Replace outliers with median value for each column
            outlier_mask = (df_med[col] < lower_bound) | (df_med[col] > upper_bound)
            df_med.loc[outlier_mask, col] = median
        print('-----------------------------------Recheck Outliers with Boxplots:------------------------------------')
        self.outliers_check(df_med)
        print('-----------------------------------Recheck the Distribution with Histograms:-----------------------')
        self.distribution_con(df_med)

        print('=================================Apply Box-Cox transformation to each numeric column:==========================')
        from scipy.stats import boxcox
        df_boxcox = input_df.copy()
        small_positive_value = 0.001    
        for col in self.list_continuous:
            # to avoid the fact that the negative values will be converted to null values in the log1p(), firstly replace the original non-positive values to a very small positive value
            df_boxcox[col] = df_boxcox[col].apply(lambda x: small_positive_value if x <= 0 else x)
            # apply box-cox
            df_boxcox[col], _ = boxcox(df_boxcox[col])
        print('-----------------------------------Recheck Outliers with Boxplots:------------------------------------')
        self.outliers_check(df_boxcox)
        print('-----------------------------------Recheck the Distribution with Histograms:-----------------------')
        self.distribution_con(df_boxcox)

        print('=================================Apply Yeo-Johnson transformation to each numeric column:==========================')
        from scipy.stats import yeojohnson
        df_yeojohnson = input_df.copy()
        small_positive_value = 0.001    
        for col in self.list_continuous:
            # to avoid the fact that the negative values will be converted to null values in the log1p(), firstly replace the original non-positive values to a very small positive value
            df_yeojohnson[col] = df_yeojohnson[col].apply(lambda x: small_positive_value if x <= 0 else x)
            # apply yeojohnson
            df_yeojohnson[col], _ = yeojohnson(df_yeojohnson[col])
        print('-----------------------------------Recheck Outliers with Boxplots:------------------------------------')
        self.outliers_check(df_yeojohnson)
        print('-----------------------------------Recheck the Distribution with Histograms:-----------------------')
        self.distribution_con(df_yeojohnson)

        print('=================================Apply Square Root transformation to each numeric column:==========================')
        from scipy.stats import yeojohnson
        df_sqr = input_df.copy()
        small_positive_value = 0.001    
        for col in self.list_continuous:
            # to avoid the fact that the negative values will be converted to null values in the log1p(), firstly replace the original non-positive values to a very small positive value
            df_sqr[col] = df_sqr[col].apply(lambda x: small_positive_value if x <= 0 else x)
            # apply square root
            df_sqr[col] = np.sqrt(df_yeojohnson[col])
        print('-----------------------------------Recheck Outliers with Boxplots:------------------------------------')
        self.outliers_check(df_yeojohnson)
        print('-----------------------------------Recheck the Distribution with Histograms:-----------------------')
        self.distribution_con(df_yeojohnson)

        print('=========================================================================================================================')
        user_input = input('Please make the decision for the method and type: 1 for winsor, 2 for median, 3 for boxcox, 4 for yeojohnson, 5 for Square Root')
        if user_input.lower() in ['1','winsor','winsorization'] :
            df_imputed = df_winsor
            chosen = 'Winsorisation'
        elif user_input.lower() in ['2','median'] :
            df_imputed = df_med
            chosen = 'Imputation with Median'
        elif user_input.lower() in ['3','boxcox','box-cox'] :
            df_imputed = df_boxcox
            chosen = 'Box-Cox Transformation'
        elif user_input.lower() in ['4','yeojohnson','yeo-johnson'] :
            df_imputed = df_yeojohnson
            chosen = 'Yeo-Johnson Transformation'
        elif user_input.lower() in ['5','square root','square-root','squareroot'] :
            df_imputed = df_sqr
            chosen = 'Square Root Transformation'
        print(f'You\'ve chosen {chosen}')
        return df_imputed

        
    
    def skewness_con(self, input_df) :
        print('==========================Check skewness of Continuous variables=====================================')
        print('Acceptable range of skewness: [-0.5, 0.5]')
        # create a list for continuous variables that have skewness not within [-0.5, 0.5]
        skew = []
        for col in self.list_continuous:
            skewness = self.df[col].skew()
            if skewness < -0.5 or skewness > 0.5 :
                skew.append(col)
            print(col,skewness)
            print('--------------------------------------')
        print('============================Continuous variables whose distribution are skewed:=========================')
        print(skew)
        print('============================Handling the skewness using np.log1p========================================')
        df_skew = input_df.copy()
        small_positive_value = 0.001
        for col in skew :
            # to avoid the fact that the negative values will be converted to null values in the log1p(), firstly replace the original non-positive values to a very small positive value
            df_skew[col] = df_skew[col].apply(lambda x: small_positive_value if x <= 0 else x)
            df_skew[col] = np.log1p(df_skew[col])
        print('Done')
        print('=============================Rechecking the skewness after reworking=======================================')
        skew_fail = [] #a list of columns has been unsuccessfully addressing using np.log1p
        for col in skew: #recheck the skewness of those skewed columns in the list skew
            skewness_rework = df_skew[col].skew()
            if skewness_rework < -0.5 or skewness_rework > 0.5 :
                skew_fail.append(col)
                print(f'{col} is still skewed: {skewness_rework}! It need another treatment!')
            else: 
                print(f'{col} has been SUCCESSFULLY reworked: {skewness_rework}')
                print(f'Replacing {col}\'s original values with the reworked values')
                input_df[col] = df_skew[col]
        if skew_fail :
            print(f'====================Applying another method: SQUARED ROOT to {skew_fail}====================================')
            for col in skew_fail :
                # reset the column before applying new method:
                df_skew[col] = input_df[col]
                # Firstly, replace the negatives and zeros with a very small positive value
                df_skew[col] = df_skew[col].apply(lambda x: small_positive_value if x <= 0 else x)
                df_skew[col] = np.sqrt(df_skew[col])
                skew_fail.remove(col)
                skewness_rework_again = df_skew[col].skew()
                # checking
                if skewness_rework_again >= -0.5 and skewness_rework_again <= 0.5 :
                    print(f'{col} has been SUCCESSFULLY reworked: {skewness_rework_again}')
                    print(f'Replacing {col}\'s original values with the reworked values')
                    input_df[col] = df_skew[col]
                    print('Done')
                else : 
                    print(f'{col} has been UNSUCCESSFULLY reworked: {skewness_rework_again}')
        elif not skew_fail : 
            print('NO COLUMN IS STILL SKEWED AFTER BEING REWORKED')
        print('DONE ADDRESSING SKEWNESS!!!!!!!')
        return input_df

#  create a column of user_id. Each row of the data is supposed to represent a set of information of a unique patient.

    def create_user_id(self, input_df):
        # Generate unique random user IDs
        num_rows = input_df.shape[0]
        unique_user_ids = np.random.choice(range(1, num_rows*10), num_rows, replace=False)

        # Add the user_id column to the DataFrame as the first column
        input_df.insert(0, 'UserID', unique_user_ids)
        return input_df



# =======================================================Working with Mysql Database======================================================================

class DB_MySQL :
    
    def __init__(self) :
        from sqlalchemy import create_engine
        import pymysql
        # Database credentials
        user = input('username:')
        password = input('password:')
        host = 'localhost'
        database = 'heart'

        # Establish pymysql connection
        connection = pymysql.connect(host=host, user=user, password=password)
        # Create database if not exists
        try:
            with connection.cursor() as cursor:
                cursor.execute("CREATE DATABASE IF NOT EXISTS heart;")
            connection.commit()
        finally:
            connection.close()

        # Reconnect to the newly created (or existing) database
        self.engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')
        self.connection = self.engine.connect()


    def import_to_mysql(self):
        from sqlalchemy import text
        table_check_query = """SELECT table_name
FROM information_schema.tables
WHERE 1=1
AND table_schema = 'heart'
AND table_name = 'HEART_DATA_CLEANED'
"""
        result = self.connection.execute(text(table_check_query))

        if result.fetchone() is None:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS heart_data_cleaned (
            UserID INT,
            Age FLOAT,
            Sex INT,
            ChestPainType INT,
            RestingBP FLOAT,
            Cholesterol FLOAT,
            FastingBS INT,
            RestingECG INT,
            MaxHR FLOAT,
            ExerciseAngina INT,
            Oldpeak FLOAT,
            ST_Slope INT,
            CountMajorVessels INT,
            Thalassemia INT,
            HeartDisease INT,
            Area INT
            );
            """
            self.connection.execute(text(create_table_query))
            print("Table 'heart_data_cleaned' created.")
        else:
            print("Table 'heart_data_cleaned' already exists.")
        data = input('what data are you importing: a dataframe or a csv file?')
        
        import inspect
        if data == 'dataframe':
            input_df_name = input('Tell me the name of the dataframe: ')
            
            # Get the current global and local variables
            current_globals = inspect.currentframe().f_back.f_globals
            current_locals = inspect.currentframe().f_back.f_locals
            
            # Check both global and local scopes for the DataFrame
            if input_df_name in current_globals and isinstance(current_globals[input_df_name], pd.DataFrame):
                input_df = current_globals[input_df_name]
            elif input_df_name in current_locals and isinstance(current_locals[input_df_name], pd.DataFrame):
                input_df = current_locals[input_df_name]
            else:
                print(f"No DataFrame found with the name '{input_df_name}'.")
                return
        elif data == 'csv' :
            input_csv = input('tell me the full name of your file')
            try:
                input_df = pd.read_csv(input_csv)
            except FileNotFoundError:
                print(f"No file found with the name '{input_csv}'.")
                return
        else:
            print("Invalid input. Please enter 'dataframe' or 'csv'.")
            return
        
        # Truncate the table before inserting new data
        truncate_query = "TRUNCATE TABLE heart_data_cleaned;"
        self.connection.execute(text(truncate_query))
        
        # Create the table if it doesn't exist
        self.connection.execute(text(table_check_query))

        # Import the DataFrame to MySQL
        input_df.to_sql('heart_data_cleaned', con=self.engine, if_exists='append', index=False)

        print("Data imported successfully.")
       
    # def Oracle_procedure(self):


# ===============================================================Oracle============================================================

class DB_Oracle:
    
    def __init__(self) :
        import pyodbc
        # Database credentials
        username = input('username:')
        password = input('password:')
        host = 'localhost'
        port = 1521
        service_name  = 'mypdb'
        conn_str = f"Driver={{Oracle in OraDB21Home1}};DBQ={host}:{port}/{service_name};UID={username};PWD={password}"
        self.connection = pyodbc.connect(conn_str)
        self.cursor = self.connection.cursor()
        print("Successfully connected to the database.")


    def import_to_oracle(self):
        table_check_query = """
        SELECT table_name FROM user_tables WHERE table_name = 'HEART_DATA_CLEANED'
        """
        self.cursor.execute(table_check_query)
        result = self.cursor.fetchone()

        if result is None:
            create_table_query = """
            CREATE TABLE heart_data_cleaned (
            UserID INT,
            Age FLOAT,
            Sex INT,
            ChestPainType INT,
            RestingBP FLOAT,
            Cholesterol FLOAT,
            FastingBS INT,
            RestingECG INT,
            MaxHR FLOAT,
            ExerciseAngina INT,
            Oldpeak FLOAT,
            ST_Slope INT,
            CountMajorVessels INT,
            Thalassemia INT,
            HeartDisease INT,
            Area INT
            );
            """
            self.cursor.execute(create_table_query)
            print("Table 'heart_data_cleaned' created.")
        else:
            print("Table 'heart_data_cleaned' already exists.")
        
        data = input('what data are you importing: a dataframe or a csv file?')
        
        import inspect
        if data == 'dataframe':
            input_df_name = input('Tell me the name of the dataframe: ').strip()
            
            # Get the current global and local variables
            current_globals = inspect.currentframe().f_back.f_globals
            current_locals = inspect.currentframe().f_back.f_locals
            
            # Check both global and local scopes for the DataFrame
            if input_df_name in current_globals and isinstance(current_globals[input_df_name], pd.DataFrame):
                input_df = current_globals[input_df_name]
            elif input_df_name in current_locals and isinstance(current_locals[input_df_name], pd.DataFrame):
                input_df = current_locals[input_df_name]
            else:
                print(f"No DataFrame found with the name '{input_df_name}'.")
                return
        elif data == 'csv' :
            input_csv = input('tell me the full name of your file with path')
            try:
                input_df = pd.read_csv(input_csv)
            except FileNotFoundError:
                print(f"No file found with the name '{input_csv}'.")
                return
        else:
            print("Invalid input. Please enter 'dataframe' or 'csv'.")
            return
        
      
        

       # Prepare columns and values placeholders for SQL INSERT
        columns = ", ".join(input_df.columns)
        values = ", ".join([":" + str(i+1) for i in range(len(input_df.columns))])
        insert_query = f"INSERT INTO heart_data_cleaned ({columns}) VALUES ({values})"

        # Insert each row of the dataframe into the Oracle table
        for _, row in input_df.iterrows():
            self.cursor.execute(insert_query, tuple(row))

        # Commit the transaction
        self.connection.commit()

        print("Data imported successfully.")





    