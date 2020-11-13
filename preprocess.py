import pandas as pd
import numpy as np
import re
import pickle
import swifter
import time
import os
from sklearn.model_selection import train_test_split

TC_NUM_CLASSES = {
    'yelp': 5,
    'yahoo': 10,
    'amazon': 5,
    'agnews': 4,
    'dbpedia': 14,
    'reddit' : 10
}
# dataset order for text classification
TC_ORDER = {
    1: ['yelp', 'agnews', 'dbpedia', 'amazon', 'yahoo', 'reddit'],
    2: ['dbpedia', 'yahoo', 'reddit','agnews', 'amazon', 'yelp'],
    3: ['yelp', 'reddit', 'yahoo', 'amazon', 'dbpedia', 'agnews'],
    4: ['reddit','agnews', 'yelp', 'amazon', 'yahoo', 'dbpedia']
}
# dataset order for question answering
# QA Not implemented
# QA_ORDER = {
#     1: ['quac', 'trweb', 'trwik', 'squad'],
#     2: ['squad', 'trwik', 'quac', 'trweb'],
#     3: ['trweb', 'trwik', 'squad', 'quac'],
#     4: ['trwik', 'quac', 'trweb', 'squad']
# }
INDIVIDUAL_CLASS_LABELS = {
    'yelp': {1: '1', 2: '2', 3: '3', 4: '4', 5: '5'},
    'dbpedia': {1: 'Company', 2: 'EducationalInstitution', 3: 'Artist',
                4: 'Athlete', 5: 'OfficeHolder', 6: 'MeanOfTransportation', 7: 'Building',
                8: 'NaturalPlace', 9: 'Village', 10: 'Animal', 11: 'Plant', 12: 'Album',
                13: 'Film', 14: 'WrittenWork'},
    'yahoo': {1: 'Society & Culture', 2: 'Science & Mathematics', 3: 'Health',
              4: 'Education & Reference', 5: 'Computers & Internet', 6: 'Sports',
              7: 'Business & Finance', 8: 'Entertainment & Music',
              9: 'Family & Relationships', 10: 'Politics & Government'},
    'amazon': {1: '1', 2: '2', 3: '3', 4: '4', 5: '5'},
    'agnews': {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}, 
    'reddit': {1: 'AskReddit', 2: 'pics', 3: 'gaming', 4: 'reddit.com', 5: 'funny',
               6: 'IAmA', 7: 'fffffffuuuuuuuuuuuu', 8: 'politics', 9: 'atheism', 10: 'trees'}
}


def preprocess(text):
    """
    Preprocesses the text
    """
    text = text.lower()
    # removes '\n' present explicitly
    text = re.sub(r"(\\n)+", " ", text)
    # removes '\\'
    text = re.sub(r"(\\\\)+", "", text)
    # removes unnecessary space
    text = re.sub(r"(\s){2,}", u" ", text)
    # replaces repeated punctuation marks with single punctuation followed by a space
    # e.g, what???? -> what?
    text = re.sub(r"([.?!]){2,}", r"\1", text)
    # appends space to $ which will help during tokenization
    text = text.replace(u"$", u"$ ")
    # replace decimal of the type x.y with x since decimal digits after '.' do not affect, e.g, 1.25 -> 1
    text = re.sub(r"(\d+)\.(\d+)", r"\1", text)
    # removes hyperlinks
    text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "", text)
    # Truncating the content after 1280 characters
    # 1280 = 128 (seq length) * 10((assumed avg. word size) 8 + (spaces on both sides) 2 = 10))
    # Note: our model uses sequences of length 128
    text = text[:1280]
    return str(text)


def create_ordered_tc_data(order, base_location=os.path.join('data','original_data'), save_location=os.path.join
('data','ordered_data'), split='train'):
    """creates ordered dataset for text classification with a maximum of 115,000 sequences
    and 7,600 sequences from each individual dataset for train and test data respectively
    i.e.,the size of the smallest training and test sets

    Args:
        order (int): which order of data to generate from TC_ORDER      
        base_location (str, optional): location of data to order. 
            Defaults to os.path.join('data','original_data')
        save_location (str, optional): location to save ordered data. 
            Defaults to os.path.join('data','ordered_data').
        TODO: This parmeter is not used 
        split (str, optional): generate test or train set. Defaults to 'train'.
    """


    if not os.path.exists(save_location):
        os.mkdir(save_location)
        os.mkdir(os.path.join(save_location, 'test'))
        os.mkdir(os.path.join(save_location, 'train'))
    dataset_sequence = TC_ORDER[order]
    ordered_dataset = {'labels': [], 'content': []}
    test_set = {'labels': [], 'content': []}
    num_classes = -1
    train_samples = 11500
    test_samples = 7600
    label_to_class = dict()
    amazon_done = False
    amazon_labels = dict()
    yelp_done = False
    yelp_labels = dict()

    for data in dataset_sequence:

        if data == 'yelp':
            yelp_done = True
            df = pd.read_csv(os.path.join(base_location, split, data +'.csv'), 
                             header=None, names=['labels', 'content'])
            df.dropna(subset=['content'], inplace=True)
            df.loc[:, 'content'] = df.content.swifter.apply(preprocess)
            if amazon_done:
                df.loc[:, 'labels'] = df.labels.swifter.apply(
                    lambda x: amazon_labels[x])
                for k, v in INDIVIDUAL_CLASS_LABELS[data].items():
                    new_key = amazon_labels[k]
                    label_to_class[new_key] = v
            else:
                df.loc[:, 'labels'] = df.labels + num_classes
                for k, v in INDIVIDUAL_CLASS_LABELS[data].items():
                    new_key = k + num_classes
                    label_to_class[new_key] = v
                    yelp_labels[k] = new_key
                num_classes += TC_NUM_CLASSES[data]

        elif data == 'amazon':
            amazon_done = True
            df = pd.read_csv(os.path.join(base_location, split, data +'.csv'),
                             header=None, names=['labels', 'title', 'content'])
            df.dropna(subset=['content'], inplace=True)
            df.loc[:, 'content'] = df.content.swifter.apply(preprocess)
            if yelp_done:
                df.loc[:, 'labels'] = df.labels.swifter.apply(
                    lambda x: yelp_labels[x])
                for k, v in INDIVIDUAL_CLASS_LABELS[data].items():
                    new_key = yelp_labels[k]
                    label_to_class[new_key] = v
            else:
                df.loc[:, 'labels'] = df.labels + num_classes
                for k, v in INDIVIDUAL_CLASS_LABELS[data].items():
                    new_key = k + num_classes
                    label_to_class[new_key] = v
                    amazon_labels[k] = new_key
                num_classes += TC_NUM_CLASSES[data]


        elif data == 'yahoo':
            df = pd.read_csv(os.path.join(base_location, split, data +'.csv'),
                             header=None, names=['labels', 'title', 'content', 'answer'])
            df.dropna(subset=['content'], inplace=True)
            print(df['labels'].head())
            df.loc[:, 'labels'] = df.labels + num_classes
            df.loc[:, 'content'] = df.content.swifter.apply(preprocess)
            # Mapping new labels to classes
            for k, v in INDIVIDUAL_CLASS_LABELS[data].items():
                new_key = k + num_classes
                label_to_class[new_key] = v
            num_classes += TC_NUM_CLASSES[data]



        elif data=='reddit':
            # dataset is agnews or dbpedia
            df = pd.read_csv(base_location+'/'+split+'/'+data+'.csv',
                             header=None, names=['labels','content'])
            df.dropna(subset=['content'], inplace=True)
            df.loc[:, 'labels'] = df.labels + num_classes
            df.loc[:, 'content'] = df.content.swifter.apply(preprocess)
            # Mapping new labels to classes
            for k, v in INDIVIDUAL_CLASS_LABELS[data].items():
                new_key = k + num_classes
                label_to_class[new_key] = v
            num_classes += TC_NUM_CLASSES[data]

        
        else:
            # dataset is agnews or dbpedia
            df = pd.read_csv(base_location+'/'+split+'/'+data+'.csv',
                             header=None, names=['labels', 'title', 'content'])
            df.dropna(subset=['content'], inplace=True)
            df.loc[:, 'labels'] = df.labels + num_classes
            df.loc[:, 'content'] = df.content.swifter.apply(preprocess)
            # Mapping new labels to classes
            for k, v in INDIVIDUAL_CLASS_LABELS[data].items():
                new_key = k + num_classes
                label_to_class[new_key] = v
            num_classes += TC_NUM_CLASSES[data]



        # filter rows with length greater than 20 (2 words including spaces on average)
        df.drop(df[df['content'].map(len) < 20].index, inplace=True)
        train, test = train_test_split(df, stratify=df['labels'], 
                                        train_size=train_samples,
                                        test_size=test_samples)
        ordered_dataset['labels'].extend(list(train.labels))
        ordered_dataset['content'].extend(list(train.content))
        test_set['labels'].extend(list(test.labels))
        test_set['content'].extend(list(test.content))

    ordered_dataframe = pd.DataFrame(ordered_dataset)

    # Shuffle the rows of the dataframe since the dataframe created has similar data grouped


    save_path = os.path.join(save_location, split, str(order)+'.csv')
    ordered_dataframe.to_csv(save_path, index=False)

    
    ordered_dataframe = pd.DataFrame(test_set)
    # Shuffle the rows of the dataframe since the dataframe created has similar data groupe
    ordered_dataframe.sample(frac=1).reset_index(drop=True, inplace=True)
    save_path = os.path.join(save_location, 'test', str(order)+'.csv')
    ordered_dataframe.to_csv(save_path, index=False)


    pkl_path = os.path.join(save_location, split, str(order)+'.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(label_to_class, f)


if __name__ == "__main__":


    # create ordered dataset
    total_time = 0
    total_order = 4
    print("Started generating training data")
    for i in range(0,total_order):
        print("Started for order {}".format(i+1))
        start = time.time()
        create_ordered_tc_data(i+1, split='train')
        end = time.time()
        print("Time taken for order {} : {} minutes".format(i+1, (end-start)/60))
        total_time += (end-start)/60
    print("Total time taken: {} for generating training and test data".format(total_time))
    print("Completed preprocessing :)")
