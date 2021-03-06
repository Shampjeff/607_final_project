import re
import pandas as pd
import operator 

# Add 'datatype' column that indicates if the record is original wiki answer as 0, training data 1, test data 2, onto 
# the dataframe - uses stratified random sampling (with seed) to sample 


# way too many inputs :(
def create_datatype(df, train_value, test_value, datatype_var, 
                    compare_dfcolumn, operator_of_compare, 
                    value_of_compare, sampling_number, sampling_seed):
    
    #subset on conditional selection
    df_subset = df[operator_of_compare(df[compare_dfcolumn], value_of_compare)]
    df_subset = df_subset.drop(columns = [datatype_var])
    
    # Sets all datatype to value for training for df_subset
    df_subset.loc[:, datatype_var] = train_value
    
    # Performs stratified random sample
    df_sampled = df_subset.groupby(
        ['Task', compare_dfcolumn], 
        group_keys=False).apply(lambda x: x.sample(min(len(x),
                                                      sampling_number), 
                                                      random_state = sampling_seed))
    df_sampled = df_sampled.drop(columns = [datatype_var])
    df_sampled.loc[:, datatype_var] = test_value
    
    for index in df_sampled.index: 
        
        # Labels all datatype_var columns with test_value for sample
        df_subset.loc[index, datatype_var] = test_value

    for index in df_subset.index:
        # Labels all datatype_var columns in df with train_value/test_value 
        df.loc[index, datatype_var] = df_subset.loc[index, datatype_var]
    
def train_test_dataframe(clean_df, random_seed=100):
    
    new_df = clean_df.copy()
    new_df.loc[:,'Datatype'] = 0

    # Creates test & training datatypes for plagiarized answers (1,2,3)
    create_datatype(new_df, 1, 2, 'Datatype', 
                    'Category', operator.gt, 0, 1, random_seed)

    # Creates test & training datatypes for NON-plagiarized answers (0)
    create_datatype(new_df, 1, 2, 'Datatype', 
                    'Category', operator.eq, 0, 2, random_seed)
    
    mapping = {0:'orig', 1:'train', 2:'test'} 
    new_df.Datatype = [mapping[item] for item in new_df.Datatype] 

    return new_df


# helper function for pre-processing text given a file
def process_file(file):
    all_text = file.read().lower()
    all_text = re.sub(r"[^a-zA-Z0-9]", " ", all_text)
    all_text = re.sub(r"\t", " ", all_text)
    all_text = re.sub(r"\n", " ", all_text)
    all_text = re.sub("  ", " ", all_text)
    all_text = re.sub("   ", " ", all_text)
    
    return all_text


def create_text_column(df, file_directory='data/'):
    '''
    Joins df to is associated `Text`. 
       :param df: A dataframe of file information
       :param file_directory: the main directory where files are stored
       
       :return: A dataframe with processed text 
    '''
    text_df = df.copy()
    text = []
    
    for row_i in df.index:
        filename = df.iloc[row_i]['File']
        file_path = file_directory + filename
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:

            # clean text using helper function
            file_text = process_file(file)
            text.append(file_text)
    
    text_df['Text'] = text
    
    return text_df
