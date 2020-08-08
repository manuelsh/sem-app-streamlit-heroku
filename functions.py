import pandas as pd
import numpy as np
import re
from numpy import dot
from numpy.linalg import norm
import re
import copy 
import time
from multiprocessing import Pool
from fasttext import train_unsupervised
import streamlit as st
from functools import partial
import warnings
import base64
import os
import io
from scipy import spatial
from scipy.spatial import distance
# from text_processing import *
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn 

#--- Functions

@st.cache(suppress_st_warning=True)
def create_model(df , column = 'Keyword'):
    st.write("Training the model")
    with open('output.txt', 'w') as f:
        f.write(df[column].str.cat(sep='\n'))
    model = train_unsupervised('output.txt', epoch = 1000 )
    os.remove('output.txt')
    return model

@st.cache(suppress_st_warning=True)
def pipeline(dataset,
            number_of_clusters = 800,
            number_of_kw = 10,
            similarity_clusters = 0.95,
            similarity_categories = 0.9,
            create_embedding_dataset = True,
            categories = False):
    
    if create_embedding_dataset:
        dataset = create_embedding_parallel(dataset , keywords_column)
        product_queries = clean_embedding(dataset)
    else:
        product_queries = dataset


    st.write('Number of Keywords to group' , len(product_queries))
    st.write('Assigning Clusters')
    st.write('Similarity Threshold: ' , similarity_clusters)
    st.write('Number of Ad groups wanted: ' , number_of_clusters)
    
    product_queries = product_queries.drop_duplicates(subset = keywords_column, 
                         keep = False)
    product_queries = product_queries.reset_index( drop = True)        
    clusters_df , rest_df = making_clusters(product_queries ,
                                            number_of_clusters = number_of_clusters ,
                                            similarity_threshold=similarity_clusters , 
                                            number_of_kw = number_of_kw)
    
    results =  clusters_df
    st.write('Number of keywords grouped first grouping',len(clusters_df))
    st.write('Number of keywords NOT grouped',len(rest_df))
    return results , rest_df


def save_result(data_set , RESULT_FILE_NAME =  "ad_groups.xlsx", categories = False):
        if 'embedding_average' in data_set.columns:
            data_set = data_set.drop(columns='embedding_average')
        data_set = data_set.sort_values(by='Ad_group_number' )
        if categories:
            data_set = data_set[['Keyword','Ad_group_name','Ad_group_number','sub_category' , 'Volume']] 
        else:
            data_set = data_set[['Keyword','Ad_group_name','Ad_group_number', 'Volume']] 
        
        data_set.to_excel("./output"+ '/' + RESULT_FILE_NAME
                                , index=False,) 

def parallelize(data, func, num_of_processes=12):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def run_on_subset(func, data_subset):
    return data_subset.apply(func)
@st.cache
def parallelize_on_rows(data, func, num_of_processes=12):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)

# def get_distances_parallel():
# 

def making_clusters(data_set, number_of_clusters ,  similarity_threshold  , number_of_kw):
    # print('Making clusters')
    
    data_set_mutable = copy.copy(data_set)
    result_keyword = pd.DataFrame()
    clusters = pd.DataFrame()

    bar = st.progress(0)
    latest_iteration = st.empty()
    for i in range(number_of_clusters):
        bar.progress(int((i+1)*100/number_of_clusters))
        latest_iteration.text(f'Iteration {i+1}')
        if len(data_set_mutable)==0:
            st.write('Breaking on first grouping , you have groupped all the KW')
            break
        distancias = 1 - distance.cdist([data_set_mutable.iloc[0]['embedding_average'].tolist()]
                                    , data_set_mutable['embedding_average'].tolist()
                                        , 'cosine')
                                
        distances =pd.Series( distancias.tolist()[0])

        distances = distances[distances >= similarity_threshold]
        sorted_distances = (distances.sort_values(ascending=False))
        indices = sorted_distances[0:number_of_kw].index

        result_keyword =  data_set_mutable.loc[indices]
        result_keyword['Ad_group_name'] = data_set_mutable.iloc[0][keywords_column]
        result_keyword['Ad_group_number'] = i
        data_set_mutable = data_set_mutable.drop(indices)
        data_set_mutable = data_set_mutable.reset_index(drop=True)

        clusters = clusters.append(result_keyword, ignore_index = True)
    
    return clusters , data_set_mutable

def average_over_terms(sentence):
    embbeding = []
    for word in sentence.split(' '):
        embbeding.append(model[word])
    return np.array(embbeding).mean(0)


def create_embedding(data_set):
    data_set['embedding_average'] =  data_set[keywords_column].apply(average_over_terms)
    return data_set

@st.cache
def create_embedding_parallel(data_set , column ):
    data_set['embedding_average'] =  parallelize_on_rows(data_set[column] ,average_over_terms )
    return data_set 

def clean_embedding(data_set):
    data_set = data_set[ ~data_set['embedding_average'].isnull()]
    return data_set

def cos_sim(a, b):
    # return 1 - spatial.distance.cosine(a, b)
    return dot(a, b)/(norm(a)*norm(b))
 

def load_data(data_file):
    data = pd.read_excel(data_file)
    return data
# @st.cache   
def show_df(df ):
     st_ms = st.multiselect("Columns",df.columns.tolist(), default= df.columns.tolist()[:1] , key = str(df))
     st.dataframe(df[st_ms])


def get_table_download_link(df, file_name):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    if 'embedding_average' in df.columns:
        df = df.drop(columns='embedding_average')
    # df = results_output.drop(columns='embedding_average')
    # csv = df.to_csv(index=False)
    # b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    # href = f'<a href="data:file/csv;base64,{encoded}">Download Excel File</a> (right-click and save as &lt;some_name&gt;.csv)'
    # href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
    towrite = io.BytesIO()
    df.to_excel(towrite,index = False,  encoding = 'UTF-8')  # write to BytesIO buffer
    towrite.seek(0)  # reset pointer
    encoded = base64.b64encode(towrite.read()).decode()  # encoded object
    href = f'<a href="data:file/csv;base64,{encoded}" download ="{file_name}">Download Excel File</a> (right-click and save as &lt;some_name&gt;.csv)'
    st.markdown(href, unsafe_allow_html=True)



@st.cache(suppress_st_warning=True)
def pipeline_exhaustive(dataset,
             keywords_column , 
            number_of_kw_min = 3,
            number_of_kw_max = 20,
            max_similarity_clusters = 0.95,
            create_embedding_dataset = True,
            categories = False):
    
    if create_embedding_dataset:
        product_queries = create_embedding_parallel(dataset , keywords_column)
    else:
        product_queries = dataset
        
    clusters_df  =   making_clusters_exhaustively(product_queries, 
                                 keywords_column, 
                                 max_similarity_clusters, 
                                 number_of_kw_min, 
                                 number_of_kw_max)
    
  
    results =  clusters_df
    return results 

def making_clusters_exhaustively(data_set, 
                                 keywords_column, 
                                 similarity_threshold = 0.95  , 
                                 number_of_kw_min = 3 , 
                                 number_of_kw_max = 20):
    
    data_set_mutable = copy.copy(search_terms_df)
    result_keyword = pd.DataFrame()
    clusters = pd.DataFrame()
    not_gruped_keyword = pd.DataFrame()
    not_gruped_df = pd.DataFrame()

    
    i = 1
    iteration = 1
    while True:

        if len(data_set_mutable)==0:

            if len(not_gruped_df)<number_of_kw_min*30:
                break

            similarity_threshold = similarity_threshold - 0.01
            if similarity_threshold <= 0.75:
                break

            iteration = iteration + 1

            data_set_mutable = data_set_mutable.append(not_gruped_df, ignore_index = True)
            data_set_mutable = data_set_mutable.reset_index(drop=True)
            not_gruped_df = pd.DataFrame() 
            st.write("new dataset len" , len(data_set_mutable), "iteration = " , iteration)

        distancias = 1 - distance.cdist([data_set_mutable.iloc[0]['embedding_average'].tolist()]
                                    , data_set_mutable['embedding_average'].tolist()
                                        , 'cosine')


        distances =pd.Series( distancias.tolist()[0])
        distances = distances[distances >= similarity_threshold]
        sorted_distances = (distances.sort_values(ascending=False))
        indices = sorted_distances[0:number_of_kw_max].index


        if len(indices) < number_of_kw_min:
            #tomo las kw y las guardo en un df
    #             indice = sorted_distances.index
            not_gruped_keyword =  data_set_mutable.loc[indices]
            #elimino las kw del dataframe actual para seguir agrupando luego
            data_set_mutable = data_set_mutable.drop(indices)
            data_set_mutable = data_set_mutable.reset_index(drop=True)
            # guardo las kw no agrupadas en el df que despues voy a querer recorrer de nvo
            not_gruped_df = not_gruped_df.append(not_gruped_keyword, ignore_index = True)

            continue





        result_keyword =  data_set_mutable.loc[indices]
        result_keyword['Ad_group_name'] = data_set_mutable.iloc[0][keywords_column]
        result_keyword['Ad_group_number'] = i
        result_keyword['iteration'] = iteration
        data_set_mutable = data_set_mutable.drop(indices)
        data_set_mutable = data_set_mutable.reset_index(drop=True)

        clusters = clusters.append(result_keyword, ignore_index = True)
        i = i + 1
        if int(i)%int(200) == int(0):
            st.write("ad group number " , i)
            
    return clusters