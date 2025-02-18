import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    - Takes inputs as two CSV files
    - Imports them as pandas dataframe.
    - Merges them into a single dataframe
    Args:
    messages_file_path str: Messages CSV file
    categories_file_path str: Categories CSV file
    Returns:
    merged_df pandas_dataframe: Dataframe obtained from merging the two input\
    data
    """
    #Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = pd.merge(messages,categories, on='id')
    #df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    """Clean dataframe by removing duplicates & converting categories from strings 
    to binary values.
    
    Args:
    df: dataframe. Dataframe containing merged content of messages & categories datasets.
       
    Returns:
    df: dataframe. Dataframe containing cleaned version of input dataframe.
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    #category_colnames = categories.loc[0].apply(lambda x:x[:len(x)-2])
    category_colnames = categories.loc[0].apply(lambda x:x[:len(x)-2])
    print(category_colnames)

    # rename the columns of `categories`
    categories.columns = category_colnames

    # set each value to be the last character of the string
    for column in categories:
        categories[column] = categories[column].apply(lambda x:x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(str)
        categories[column] = pd.to_numeric(categories[column])
    
        # check unique value counts in each coulmn
        print(categories[column].value_counts())
        
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df= pd.concat([df, categories], axis = 1)

    # drop duplicates
    df.drop_duplicates(inplace= True)
    

    #Remove rows with a related value of 2 from the dataset
    df = df[df['related'] != 2]
    
    return df
def save_data(df, database_filename):
    """Save into  SQLite database.

    inputs:
    df: dataframe. Dataframe containing cleaned version of merged message and
    categories data.
    database_filename: string. Filename for output database.

    outputs:
    None
    """

    engine = create_engine('sqlite:///'+ database_filename)
    table_name = database_filename.replace(".db","") + "_table"
    print('table_name:{}'.format(table_name))
    df.to_sql(table_name, engine, index=False, if_exists='replace')
    
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print(df.nunique())
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')
        
        

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
