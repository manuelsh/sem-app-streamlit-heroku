import base64
import io
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import streamlit as st
from fasttext import train_unsupervised
from numpy import dot
from numpy.linalg import norm
from scipy.spatial import distance

#--- Functions

@st.cache_resource(show_spinner=False, max_entries=2, scope='session')
def create_model(df, column='Keyword', epoch=1000):
    """Train and cache a FastText model for the selected text column."""
    if column not in df.columns:
        raise ValueError(f"Column {column!r} is not present in the data")

    corpus = df[column].dropna().astype(str)
    if corpus.empty or not corpus.str.strip().any():
        raise ValueError("Cannot train a model from an empty text column")

    st.write("Training the model")
    corpus_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        ) as corpus_file:
            corpus_file.write(corpus.str.cat(sep="\n"))
            corpus_path = corpus_file.name
        return train_unsupervised(corpus_path, epoch=int(epoch), minCount=1)
    finally:
        if corpus_path is not None:
            try:
                os.unlink(corpus_path)
            except FileNotFoundError:
                pass

def pipeline(dataset,
            model,
            keywords_column,
            number_of_clusters = 800,
            number_of_kw = 10,
            similarity_clusters = 0.95,
            similarity_categories = 0.9,
            create_embedding_dataset = True,
            categories = False):
    
    if create_embedding_dataset:
        dataset = create_embedding_parallel(dataset, keywords_column, model)
        product_queries = clean_embedding(dataset)
    else:
        product_queries = dataset


    st.write('Number of Keywords to group' , len(product_queries))
    st.write('Assigning Clusters')
    st.write('Similarity Threshold: ' , similarity_clusters)
    st.write('Number of Ad groups wanted: ' , number_of_clusters)
    
    product_queries = product_queries.drop_duplicates(
        subset=keywords_column, keep='first'
    )
    product_queries = product_queries.reset_index( drop = True)        
    clusters_df , rest_df = making_clusters(product_queries ,
                                            keywords_column=keywords_column,
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
    """Apply ``func`` to pandas chunks without creating empty workers."""
    if len(data) == 0:
        return data.copy()

    process_count = min(
        max(1, int(num_of_processes)), len(data), os.cpu_count() or 1
    )
    boundaries = np.linspace(0, len(data), process_count + 1, dtype=int)
    data_split = [
        data.iloc[boundaries[index] : boundaries[index + 1]]
        for index in range(process_count)
    ]
    if process_count == 1:
        return pd.concat([func(data_split[0])])
    with ThreadPoolExecutor(max_workers=process_count) as executor:
        return pd.concat(executor.map(func, data_split))

def run_on_subset(func, data_subset):
    return data_subset.apply(func)
def parallelize_on_rows(data, func, num_of_processes=12):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)

# def get_distances_parallel():
# 

def making_clusters(
    data_set,
    keywords_column,
    number_of_clusters,
    similarity_threshold,
    number_of_kw,
):
    # print('Making clusters')
    
    if number_of_clusters < 1 or number_of_kw < 1:
        raise ValueError("Cluster and keyword counts must be positive")

    data_set_mutable = data_set.copy()
    clusters = data_set.iloc[0:0].copy()
    clusters['Ad_group_name'] = pd.Series(dtype='object')
    clusters['Ad_group_number'] = pd.Series(dtype='int64')

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

        result_keyword = data_set_mutable.loc[indices].copy()
        result_keyword['Ad_group_name'] = data_set_mutable.iloc[0][keywords_column]
        result_keyword['Ad_group_number'] = i
        data_set_mutable = data_set_mutable.drop(indices)
        data_set_mutable = data_set_mutable.reset_index(drop=True)

        clusters = pd.concat([clusters, result_keyword], ignore_index=True)
    
    return clusters , data_set_mutable

def average_over_terms(sentence, model):
    if pd.isna(sentence):
        return np.full(model.get_dimension(), np.nan, dtype=float)
    words = str(sentence).split()
    if not words:
        return np.full(model.get_dimension(), np.nan, dtype=float)
    embeddings = [model[word] for word in words]
    return np.asarray(embeddings).mean(axis=0)


def create_embedding(data_set, column, model):
    if column not in data_set.columns:
        raise ValueError(f"Column {column!r} is not present in the data")
    result = data_set.copy()
    result['embedding_average'] = result[column].apply(
        partial(average_over_terms, model=model)
    )
    return result

def create_embedding_parallel(data_set, column, model):
    """Create embeddings serially; FastText model objects are not process-safe."""
    return create_embedding(data_set, column, model)

def clean_embedding(data_set):
    valid = data_set['embedding_average'].apply(
        lambda value: (
            isinstance(value, np.ndarray)
            and value.size > 0
            and np.isfinite(value).all()
        )
    )
    return data_set.loc[valid].copy()

def cos_sim(a, b):
    denominator = norm(a) * norm(b)
    if denominator == 0:
        return 0.0
    return dot(a, b) / denominator


def closest_embedding_indices(embeddings, cluster_embeddings):
    """Return the nearest cluster index for each embedding by cosine similarity."""
    if len(embeddings) == 0:
        return np.array([], dtype=int)
    if len(cluster_embeddings) == 0:
        raise ValueError("At least one cluster embedding is required")

    embedding_matrix = np.vstack(embeddings)
    cluster_matrix = np.vstack(cluster_embeddings)
    if embedding_matrix.shape[1] != cluster_matrix.shape[1]:
        raise ValueError("Embedding and cluster dimensions must match")

    valid_embeddings = (
        np.isfinite(embedding_matrix).all(axis=1)
        & (np.linalg.norm(embedding_matrix, axis=1) > 0)
    )
    valid_clusters = (
        np.isfinite(cluster_matrix).all(axis=1)
        & (np.linalg.norm(cluster_matrix, axis=1) > 0)
    )
    if not valid_clusters.any():
        raise ValueError("At least one finite, non-zero cluster embedding is required")

    result = np.full(len(embedding_matrix), -1, dtype=int)
    if valid_embeddings.any():
        similarities = 1 - distance.cdist(
            embedding_matrix[valid_embeddings],
            cluster_matrix[valid_clusters],
            metric='cosine',
        )
        original_cluster_indices = np.flatnonzero(valid_clusters)
        result[valid_embeddings] = original_cluster_indices[
            np.argmax(similarities, axis=1)
        ]
    return result
 

def load_data(data_file):
    data = pd.read_excel(data_file)
    return data
# @st.cache   
def show_df(df, key="dataframe_columns"):
    selected_columns = st.multiselect(
        "Columns",
        df.columns.tolist(),
        default=df.columns.tolist()[:1],
        key=key,
    )
    st.dataframe(df[selected_columns])


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
    df.to_excel(towrite, index=False)  # write to BytesIO buffer
    towrite.seek(0)  # reset pointer
    encoded = base64.b64encode(towrite.read()).decode()  # encoded object
    href = f'<a href="data:file/csv;base64,{encoded}" download ="{file_name}">Download Excel File</a> (right-click and save as &lt;some_name&gt;.csv)'
    st.markdown(href, unsafe_allow_html=True)



def pipeline_exhaustive(dataset,
             keywords_column , 
             model=None,
            number_of_kw_min = 3,
            number_of_kw_max = 20,
            max_similarity_clusters = 0.95,
            create_embedding_dataset = True,
            categories = False):
    
    if create_embedding_dataset:
        if model is None:
            raise ValueError("A trained model is required to create embeddings")
        product_queries = create_embedding_parallel(dataset, keywords_column, model)
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
    
    data_set_mutable = data_set.copy()
    clusters = data_set.iloc[0:0].copy()
    clusters['Ad_group_name'] = pd.Series(dtype='object')
    clusters['Ad_group_number'] = pd.Series(dtype='int64')
    clusters['iteration'] = pd.Series(dtype='int64')
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

            data_set_mutable = pd.concat(
                [data_set_mutable, not_gruped_df], ignore_index=True
            )
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
            not_gruped_keyword = data_set_mutable.loc[indices].copy()
            #elimino las kw del dataframe actual para seguir agrupando luego
            data_set_mutable = data_set_mutable.drop(indices)
            data_set_mutable = data_set_mutable.reset_index(drop=True)
            # guardo las kw no agrupadas en el df que despues voy a querer recorrer de nvo
            not_gruped_df = pd.concat(
                [not_gruped_df, not_gruped_keyword], ignore_index=True
            )

            continue





        result_keyword = data_set_mutable.loc[indices].copy()
        result_keyword['Ad_group_name'] = data_set_mutable.iloc[0][keywords_column]
        result_keyword['Ad_group_number'] = i
        result_keyword['iteration'] = iteration
        data_set_mutable = data_set_mutable.drop(indices)
        data_set_mutable = data_set_mutable.reset_index(drop=True)

        clusters = pd.concat([clusters, result_keyword], ignore_index=True)
        i = i + 1
        if int(i)%int(200) == int(0):
            st.write("ad group number " , i)
            
    return clusters
