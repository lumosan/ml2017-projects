import numpy as np
import csv
import copy

class DataFrame:
    '''
    This class is used as a data-container,
    representeing column-organized information
    read from csv files.
    '''
    column_labels = {}
    data = None
    
    def __init__(self, csv_path=None):
        
        # Leave the labels and data empty if the path is None
        if csv_path is None:
            return
        
        temp_data = None
        
        with open(csv_path) as csv_file:
            csv_reader =  csv.reader(csv_file)
            n_rows = sum(1 for row in csv_reader)
        
            # Reset reader's head pointer
            csv_file.seek(0)
            
            for row_idx, row in enumerate(csv_reader):
                if (row_idx == 0):
                    
                    # Fill in dictionary with (column_name:column_index)
                    for column_idx, column_label in enumerate(row):
                        self.column_labels[column_label] = column_idx
                    temp_data = [[0 for x in range(n_rows-1)] for y in range(len(self.column_labels))]
                else:
                    
                    # Fill data in a column-oriented fashion
                    for column_idx, column_value in enumerate(row):
                        temp_data[column_idx][row_idx-1] = column_value
        
        # Store all the data into an 'ndarray'
        self.data = np.array(temp_data)
    
    # targets have to be labels, not indices
    def get_columns(self, targets=None):
        '''
        Returns a copy the desired columns' data as a 
        list of Column objects.
        '''
        columns = []
        
        if targets is not None:
            if all(isinstance(label, str) for label in targets):
                columns = [Column(label, self.data[self.column_labels[label],:]) for label in targets]
        else:
            columns = [Column(label, self.data[self.column_labels[label],:]) for label in self.column_labels]
            
        return columns
    
    # 'at' cannot have repeated values
    def insert(self, columns, at):
        '''
        Returns a new DataFrame with newly
        inserted columns, at a designated index.
        '''
        dataframe_clone = self.__clone()
        
        # Zip columns and their desired indexes together for iteration
        col_idx_zipped = sorted(zip(columns, at), key = lambda t: t[1])
        
        # Create a new dictionary of labels to append
        new_labels = dict(col_idx_zipped)
        new_labels = {column.label: idx for column, idx in col_idx_zipped}
        
        # Insert the columns' data
        for column, idx in col_idx_zipped:
            dataframe_clone.data = np.insert(dataframe_clone.data, idx, copy.deepcopy(column.values), 0)
        
        # Apply index offset where needed
        for column, idx in col_idx_zipped:
            offset_column_labels = {}
            for k, v in dataframe_clone.column_labels.items():
                offset_column_labels[k] = (v + 1) if (v >= idx) else v
            dataframe_clone.column_labels = offset_column_labels
           
        # Append the new labels to previously existing ones
        dataframe_clone.column_labels = {**dataframe_clone.column_labels, **new_labels}
        
        return dataframe_clone
    
    # target_axix=0 is columns, target_axix=1 is rows
    def drop(self, targets, target_axis=0):
        '''
        Returns a new DataFrame without the
        dropped columns/rows.
        '''
        dataframe_clone = self.__clone()
        dropable_indexes = []
        
        # All elements in the list are indexes
        if all(isinstance(index, int) for index in targets):
            dropable_indexes = targets
            dropable_keys = [key for key in dataframe_clone.column_labels if dataframe_clone.column_labels[key] in targets]
            dataframe_clone.data = np.delete(dataframe_clone.data, [dataframe_clone.column_labels.pop(label) for label in dropable_keys], axis=target_axis)
        
        # All elements in the list are labels
        elif all(isinstance(label, str) for label in targets) and target_axis == 0:
            dropable_indexes = [self.column_labels[label] for label in targets]
            dataframe_clone.data = np.delete(dataframe_clone.data, [dataframe_clone.column_labels.pop(label) for label in targets], axis=target_axis)
        
        # Non-expected parameters, just return the whole DataFrame
        else:
            return dataframe_clone
        
        # Apply index offset where needed
        dropable_indexes.sort(reverse=True)
        for idx in dropable_indexes:
            offset_column_labels = {}
            for k, v in dataframe_clone.column_labels.items():
                offset_column_labels[k] = (v - 1) if (v >= idx) else v
            dataframe_clone.column_labels = copy.deepcopy(offset_column_labels)
        
        return dataframe_clone
    
    def replace(self, existing_value, new_value):
        '''
        Replaces all occurrences of a given value,
        in the DataFrame, witha new specified value.
        '''
        dataframe_clone = self.__clone()
        if np.isnan(existing_value):
            dataframe_clone.data[np.isnan(dataframe_clone.data)] = new_value
        else:
            dataframe_clone.data[dataframe_clone.data == existing_value] = new_value
        return dataframe_clone
    
    def mean(self):
        '''
        Returns a new DataFrame with the columns'
        mean values.
        '''
        mean_df = DataFrame()
        mean_df.column_labels = copy.deepcopy(self.column_labels)
        
        columns = self.get_columns()
        
        temp_data = [[0 for x in range(1)] for y in range(len(self.column_labels))]
        for column in columns:
            temp_data[self.column_labels[column.label]][0] = column.mean()
            
        mean_df.data = np.array(temp_data)
        
        return mean_df
    
    def std(self):
        '''
        Returns a new DataFrame with the columns'
        stadard deviation values.
        '''
        std_df = DataFrame()
        std_df.column_labels = copy.deepcopy(self.column_labels)
        
        columns = self.get_columns()
        
        temp_data = [[0 for x in range(1)] for y in range(len(self.column_labels))]
        for column in columns:
            temp_data[self.column_labels[column.label]][0] = column.std()
            
        std_df.data = np.array(temp_data)
        
        return std_df
    
    def normalize(self):
        '''
        Returns a DataFrame with column-based
        normalization.
        '''
        normalized_df = DataFrame()
        normalized_df.column_labels = copy.deepcopy(self.column_labels)
        
        temp_data = [[0 for x in range(len(self.data[0,:]))] for y in range(len(self.column_labels))]
        for column in self.get_columns():
            temp_data[normalized_df.column_labels[column.label]] = column.normalize().values
        
        normalized_df.data = np.array(temp_data)
        
        return normalized_df
    
    def corr(self):
        '''
        Returns a DataFrame with the correlation
        coeficients between all of the columns.
        '''
        corr_df = DataFrame()
        corr_df.column_labels = copy.deepcopy(self.column_labels)
        corr_df.data = np.corrcoef(self.data)
        
        return corr_df
    
    def as_type(self, target_type):
        '''
        Attempts to change the DataFrame's data
        type to a single type (target_type). The
        returned DataFrame is a copy of 'self'.
        '''
        dataframe_clone = self.__clone()
        dataframe_clone.data = dataframe_clone.data.astype(target_type)
        return dataframe_clone
    
    def round_values(self, decimals=4):
        '''
        Returna a new DataFrame with all float
        rounded to the specified number of decimal
        places.
        '''
        rounded_df = self.__clone()
        rounded_df.data = np.around(rounded_df.data, decimals)
        
        return rounded_df
    
    def write_to_csv(self, file_path):
        with open(file_path, 'w', newline='') as csv_out:
            csv_writer = csv.writer(csv_out)
            sorted_labels = sorted(self.column_labels.keys(), key = lambda t: self.column_labels[t])
            csv_writer.writerow(sorted_labels)
            for i in range(len(self.data[0,:])):
                csv_writer.writerow(self.data[:,i])
    
    def __clone(self):
        '''
        Creates and returns a clone of the current
        DataFrame object (creating a deep copy of
        all its components).
        '''
        dataframe_clone = copy.deepcopy(self)
        dataframe_clone.column_labels = copy.deepcopy(self.column_labels)
        dataframe_clone.data = copy.deepcopy(self.data)
        return dataframe_clone
    
    def __repr__(self):
        '''
        Default class' representation.
        '''
        return str(self)
    
    def __str__(self):
        '''
        Default class' string representation.
        '''
        max_columns = min(6, len(self.column_labels.keys()))
        max_rows = min(8, len(self.data[0,:]))
        max_string_size = 13
        final_string = '| '
        
        # Add the schema to the top
        for idx, label in enumerate(sorted(self.column_labels.keys(), key = lambda t: self.column_labels[t])):
            if (idx < max_columns):
                label_rep = label if (len(label) <= max_string_size) else label[:max_string_size-3] + '...'
                final_string += label_rep.rjust(max_string_size) + ' | '
            elif (idx == max_columns):
                final_string += '...'
            else:
                break
        
        final_string += '\n'
        final_string += '-' * (max_string_size * max_columns + (max_columns + 1) * 3)
        
        # Add the first rows as preview
        for i in range(max_rows):
            final_string += '\n| '
            for idx, value in enumerate(self.data[:,i]):
                if (idx < max_columns):
                    value_rep = str(value) if (len(str(value)) <= max_string_size) else str(value)[:max_string_size-3] + '...'
                    final_string += value_rep.rjust(max_string_size) + ' | '
                elif (idx == max_columns):
                    final_string += '...'
                else:
                    break
        
        if (len(self.data[0,:]) > max_rows):
            final_string += '\n(...)\n'
        else:
            final_string += '\n'
        
        return final_string
    
class Column:
    '''
    This class is meant as single column's
    data representation.
    '''
    label = None
    values = None
    
    def __init__(self, label, values):
        self.label = label
        self.values = values
        
    def mean(self):
        '''
        Calculates the column's values mean,
        while ignoring NaNs.
        '''
        return np.nanmean(self.values)
    
    def std(self):
        '''
        Calculates the column's values stadrad
        deviation, while ignoring NaNs.
        '''
        return np.nanstd(self.values)
    
    def normalize(self):
        '''
        Returns a column with normalized values.
        '''
        column_clone = self.__clone()
        column_clone.values = column_clone.values - column_clone.mean()
        column_clone.values = column_clone.values / column_clone.std()
        return column_clone
    
    def nonan(self):
        '''
        Returns an Column with all the non NaN
        values.
        '''
        column_clone = self.__clone()
        column_clone.values = column_clone.values[~np.isnan(column_clone.values)]
        return column_clone
    
    def __clone(self):
        column_clone = copy.deepcopy(self)
        column_clone.label = self.label
        column_clone.values = copy.deepcopy(self.values)
        return column_clone