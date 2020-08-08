import pandas as pd
import numpy as np
import time
import streamlit as st
from multiprocessing import  Pool
from functools import partial
import base64
import re 
import os
import io
import gc

import SessionState
from text_normalization import *
from functions import *


def main():
    st.title('Adquity - SEM Campaign Builder App')

    module_program = st.sidebar.selectbox("Choose the Module", options = ["Create Ad Groups", 
                                                                          "Create Ad Groups - Exhaustive"  , 
                                                                          "Create Ad Template" , 
                                                                          "Create Ad group Features" , 
                                                                          "Autobuilder"])

    if module_program == "Create Ad Groups":
        session_state = SessionState.get(name="", button_sent=False)


        similarity_clusters = st.sidebar.slider("Choose Similarity threshold", 0.70, 0.99, 0.95)
        number_of_clusters = st.sidebar.number_input("Enter the number of Ad groups you want: ",format='%i' , value = 10)
        number_of_kw_per_adgroup = st.sidebar.number_input("Enter the maximum number of Keywords you want in an Adgroup : ",format='%i' , value = 10)





        uploaded_file = st.file_uploader("Choose a  file")
        if uploaded_file is not None:
            search_terms_df = load_data(uploaded_file)
            search_terms_columns = search_terms_df.columns
            keywords_column = st.selectbox("Which columns has the keywords in the file", options = search_terms_columns)
            volumn_column = st.selectbox("Which columns has the volume/number of clicks in the file", options = search_terms_columns)
            search_terms_df = search_terms_df.astype({keywords_column:"str"})
            search_terms_df = search_terms_df.sort_values(by = volumn_column , ascending = False )
            search_terms_df = search_terms_df.reset_index(drop=True)
            st.write("**File uploaded and read**")


            session_state.show_df_1 = st.checkbox('Show input file')
            if session_state.show_df_1:
                st.write(search_terms_df)
            session_state.cut_dataset = st.checkbox('Cut dataset?')
            if session_state.cut_dataset:
                len_dataset = st.number_input("Enter the maximum number of rows you want",format='%i' , value = 1000)
                search_terms_df = search_terms_df[:len_dataset]



            text_norm_bool = st.sidebar.checkbox("Want to normalize text?")
            if text_norm_bool:
                lenguage = st.sidebar.selectbox("Select lenguage" , ["Spanish" , "English" , "French"])
                search_terms_df['processed'] = parallelize_on_rows(search_terms_df[keywords_column],partial(preprocess,lenguage=lenguage))
                keywords_column = 'processed'




            session_state.create_model = st.checkbox('Create Model?')
            if session_state.create_model :
                if uploaded_file is  None:
                    st.warning('You have to upload a file first!')
                else:
                    model = create_model(search_terms_df, column = keywords_column )
                    st.spinner("Training the model")
                    st.write("Model trained")   


        session_state.create_ad_groups_bool = st.checkbox('Create Ad Groups?')
        if session_state.create_ad_groups_bool:
            st.write("**Creating campaign...**")
            start = time.time()
            results_output , rest_df_output = pipeline(dataset = search_terms_df ,
                                                similarity_clusters = similarity_clusters,
                                                number_of_clusters = int(number_of_clusters),
                                                    number_of_kw = int(number_of_kw_per_adgroup),
                                                    create_embedding_dataset = True )
            end = time.time()
            st.write("time: ",end - start)
            st.success("**_Done!_**")

            session_state.show_results_bool = st.checkbox('Show dataframe of results')
            if session_state.show_results_bool:
                show_df(results_output)

            session_state.download_data_bool = st.checkbox('Download Data?' , key = str(results_output))
            if session_state.download_data_bool:
                results_output = results_output.drop(columns='embedding_average') 
                # results_output = results_output[[keywords_column,"Ad_group_name",volumn_column]]
                towrite = io.BytesIO()
                results_output.to_excel(towrite,index = False,  encoding = 'UTF-16',sheet_name='Ad_groups')  # write to BytesIO buffer
                towrite.seek(0)  # reset pointer
                encoded = base64.b64encode(towrite.read()).decode()  # encoded object
                href = f'<a href="data:file/csv;base64,{encoded}" download ="Ad_groups.xlsx">Download Excel File</a> (right-click and save as &lt;some_name&gt;.csv)'
                st.markdown(href, unsafe_allow_html=True)                

    if module_program == "Create Ad Groups - Exhaustive":
        session_state = SessionState.get(name="", button_sent=False)


        # similarity_clusters = st.sidebar.slider("Choose Similarity threshold", 0.70, 0.99, 0.95)
        number_of_kw_per_adgroup_max = st.sidebar.number_input("Enter the maximum number of Keywords you want in an Adgroup : ",format='%i' , value = 20)
        number_of_kw_per_adgroup_min = st.sidebar.number_input("Enter the minimum number of Keywords you want in an Adgroup : ",format='%i' , value = 5)


        uploaded_file = st.file_uploader("Choose a  file")
        if uploaded_file is not None:
            search_terms_df = load_data(uploaded_file)
            search_terms_columns = search_terms_df.columns
            keywords_column = st.selectbox("Which columns has the keywords in the file", options = search_terms_columns)
            volumn_column = st.selectbox("Which columns has the volume/number of clicks in the file", options = search_terms_columns)
            search_terms_df = search_terms_df.astype({keywords_column:"str"})
            search_terms_df = search_terms_df.sort_values(by = volumn_column , ascending = False )
            search_terms_df = search_terms_df.reset_index(drop=True)
            st.write("**File uploaded and read**")
            session_state.cut_dataset = st.checkbox('Cut dataset?')
            if session_state.cut_dataset:
                len_dataset = st.number_input("Enter the maximum number of rows you want",format='%i' , value = 1000)
                search_terms_df = search_terms_df[:len_dataset]

            text_norm_bool = st.sidebar.checkbox("Want to normalize text?")
            if text_norm_bool:
                lenguage = st.sidebar.selectbox("Select lenguage" , ["Spanish" , "English" , "French"])
                search_terms_df['processed'] = parallelize_on_rows(search_terms_df[keywords_column],partial(preprocess,lenguage=lenguage))
                keywords_column = 'processed'


            session_state.show_df_1 = st.checkbox('Show input file')
            if session_state.show_df_1:
                st.write(search_terms_df)    

            session_state.create_model = st.checkbox('Create Model?')
            if session_state.create_model :
                if uploaded_file is  None:
                    st.warning('You have to upload a file first!')
                else:
                    model = create_model(search_terms_df, column = keywords_column )
                    st.spinner("Training the model")
                    st.write("Model trained")   



            session_state.create_ad_groups_bool = st.checkbox('Create Ad Groups?')
            if session_state.create_ad_groups_bool:
                st.write("**Creating campaign...**")
                start = time.time()
                results_output = pipeline_exhaustive(dataset = search_terms_df ,
                                                    number_of_kw_min=number_of_kw_per_adgroup_min,
                                                    number_of_kw_max=number_of_kw_per_adgroup_max,                                                 
                                                    keywords_column = keywords_column,
                                                    max_similarity_clusters = 0.99)

                end = time.time()
                st.write("time: ",end - start)
                st.success("**_Done!_**")

                session_state.show_results_bool = st.checkbox('Show dataframe of results')
                if session_state.show_results_bool:
                    show_df(results_output)


                session_state.download_data_bool = st.checkbox('Download Data?' , key = str(results_output))
                if session_state.download_data_bool:
                    results_output = results_output.drop(columns='embedding_average') 
                    # results_output = results_output[[keywords_column,"Ad_group_name",volumn_column,'Ad_group_number',"iteration"]]

                    towrite = io.BytesIO()
                    results_output.to_excel(towrite,index = False,  encoding = 'UTF-16',sheet_name='Ad_groups')  # write to BytesIO buffer
                    towrite.seek(0)  # reset pointer
                    encoded = base64.b64encode(towrite.read()).decode()  # encoded object
                    href = f'<a href="data:file/csv;base64,{encoded}" download ="Ad_groups.xlsx">Download Excel File</a> (right-click and save as &lt;some_name&gt;.csv)'
                    st.markdown(href, unsafe_allow_html=True)



    if module_program ==  "Create Ad group Features":
        st.write("Ad Creation Process")

        ad_groups_file = st.file_uploader("Upload Ad groups file")
        if ad_groups_file is not None:
            ad_groups_df = pd.read_excel(ad_groups_file) 
            st.write("**File uploaded and read**")
            if st.checkbox('Show ad groups file'):
                st.write(ad_groups_df)

        features_file = st.file_uploader("Upload features file")
        if features_file is not None:
            features_df = pd.read_excel(features_file) 

            feature_columns = features_df.columns
            for column in feature_columns:
                features_df = features_df.astype({column:"str"})
                features_df[column] = features_df[column].str.lower()

            st.write("**File uploaded and read**")

            show_data_bool = st.checkbox('Show features df')
            if show_data_bool:
                st.write(features_df)


        def assign_feature(ad_groupe_name , features):
            for feature in features:
                if feature in ad_groupe_name:
                    return feature
            return


        def assign_features(dataframe):
            for feature in feature_columns:
                feature_list = features_df[feature].unique().tolist()
                dataframe[feature] = dataframe["Ad_group_name"].apply(assign_feature ,args=(feature_list,))
            return dataframe

        if features_file is not None and ad_groups_file is not None:
            ad_groups_df = assign_features(ad_groups_df)
            ad_groups_df

        download_data_bool = st.checkbox('Download Data?' , key = "feature_extraction")
        if download_data_bool:
            towrite = io.BytesIO()
            ad_groups_df.to_excel(towrite, sheet_name='Ad_groups',index = False,  encoding = 'UTF-16')
            towrite.seek(0)  # reset pointer
            encoded = base64.b64encode(towrite.read()).decode()  # encoded object
            href = f'<a href="data:file/csv;base64,{encoded}" download ="Ad_groups_with_features.xlsx">Download Excel File</a> (right-click and save as &lt;some_name&gt;.csv)'
            st.markdown(href, unsafe_allow_html=True)  



    ############################################### AD CREATION TEMPLATE

    if module_program == "Create Ad Template":
        uploaded_file = st.sidebar.file_uploader("Choose a  file")

        if uploaded_file is not None:
            search_terms_df = pd.read_excel(uploaded_file)
            st.write("**File uploaded and read**")
            if st.checkbox('Show input file'):
                st.write(search_terms_df)

            feature_columns = search_terms_df.columns.difference(["Ad_group_name"])

    ############################################################## HEADLINES

        def create_headline(row):

            features_available = []
            text_options = list()
            long_text = hl_1_text_long.title()
            short_text = hl_1_text_short.title()

            if manual_text and include_features:
                features = features_selection 
                for feature in features:
                    if row[feature] == row[feature]:
                        features_available.append(feature)

                features = features_available

                if position == "before":
                    text_1 = long_text
                    text_options.append(text_1)
                    for feature in features:
                        new_hl = text_1 +" "+row[feature].title()
                        if len(new_hl) <= max_text_len:
                            text_1 = text_1 +" "+row[feature].title()
                            text_options.append(text_1)

                    text_1 = short_text
                    for feature in features:
                        new_hl = text_1 +" "+row[feature].title()
                        if len(new_hl) <= max_text_len:
                            text_1 = text_1 +" "+row[feature].title()
                            text_options.append(text_1)

                if position == "after":
                    text_1 = long_text
                    text_options.append(text_1)
                    for feature in features:
                        new_hl = row[feature].title() + " " + text_1 
                        if len(new_hl) <= max_text_len:
                            text_1 = new_hl
                            text_options.append(text_1)

                    text_1 = short_text
                    for feature in features:
                        new_hl = row[feature].title() + " " + text_1 
                        if len(new_hl) <= max_text_len:
                            text_1 = new_hl
                            text_options.append(text_1)

                head_line = max([text for text in text_options if len(text) <= max_text_len],key=len)

            if manual_text and not include_features: 
                head_line = long_text

            return head_line      



        HL_1 = st.sidebar.checkbox('Create Head Line 1')
        if HL_1:

            max_text_len = st.number_input("Max Length for this Head Line",format='%i' , value = 30, key="1" )

            manual_text = st.checkbox('Want To Introduce Some Manual Text????')
            if manual_text:
                position = st.selectbox("Select the position of the text" ,['before' , 'after'])
                hl_1_text_long = st.text_input('Manual text - Long option',value = "")
                hl_1_text_short = st.text_input('Manual text - Short option', value = "")
                if len(hl_1_text_long) > max_text_len:
                    st.warning("Warning, Text is too long")

            include_features = st.checkbox('Want To Introduce Features?')
            if include_features:
                features_list = feature_columns.tolist()
                features_selection = st.multiselect("Select the features, sort them by importance",features_list)

            create_hl = st.checkbox('Create HL_1?')
            if create_hl:
                search_terms_df["Head_line_1"] = search_terms_df.apply(create_headline , axis=1)

                search_terms_df

        HL_2 = st.sidebar.checkbox('Create Head Line 2')
        if HL_2:

            max_text_len = st.number_input("Max Length for this Head Line",format='%i' , value = 30 , key="2" )

            manual_text = st.checkbox('Want To Introduce Some Manual Text????', key="2")
            if manual_text:
                position = st.selectbox("Select the position of the text" ,['before' , 'after'],key="2")
                hl_1_text_long = st.text_input('Manual text - Long option',value = "",key="2")
                hl_1_text_short = st.text_input('Manual text - Short option', value = "",key="2")
                if len(hl_1_text_long) > max_text_len:
                    st.warning("Warning, Text is too long")

            include_features = st.checkbox('Want To Introduce Features?',key="2")
            if include_features:
                features_list = feature_columns.tolist()
                features_selection = st.multiselect("Select the features, sort them by importance",features_list,key="2")

            create_hl = st.checkbox('Create HL_2?')
            if create_hl:
                search_terms_df["Head_line_2"] = search_terms_df.apply(create_headline , axis=1)

                search_terms_df



        HL_3 = st.sidebar.checkbox('Create Head Line 3')
        if HL_3:

            max_text_len = st.number_input("Max Length for this Head Line",format='%i' , value = 30 , key="3" )

            manual_text = st.checkbox('Want To Introduce Some Manual Text????', key="3")
            if manual_text:
                position = st.selectbox("Select the position of the text" ,['before' , 'after'],key="3")
                hl_1_text_long = st.text_input('Manual text - Long option',value = "",key="3")
                hl_1_text_short = st.text_input('Manual text - Short option', value = "",key="3")
                if len(hl_1_text_long) > max_text_len:
                    st.warning("Warning, Text is too long")

            include_features = st.checkbox('Want To Introduce Features?',key="3")
            if include_features:
                features_list = feature_columns.tolist()
                features_selection = st.multiselect("Select the features, sort them by importance",features_list,key="3")


            create_hl = st.checkbox('Create HL_3?')
            if create_hl:
                search_terms_df["Head_line_3"] = search_terms_df.apply(create_headline , axis=1)

            search_terms_df



    ###########################################################################DESCRIPTIONS 
        def description_insert(row , description):
            features = features_selection
            features_available = []

            if include_features: 
                for feature in features:
                    if row[feature] == row[feature]:
                        features_available.append(feature)

                features = features_available

            if features == []:
                description = description.format("")
            else:
                description = description.format(row[features[0]].title())

            return description


        if st.sidebar.checkbox('Create Description 1'):
            max_text_len = st.number_input("Max Length for this description",format='%i' , value = 90 , key="3" )
            description_1 = st.text_input("Text for description 1" , value="")
            if len(description_1) > max_text_len:
                st.warning("Text is too long")

            include_features = st.checkbox('Want To Introduce Features?',key="d1")
            if include_features:
                features_list = feature_columns.tolist()
                features_selection = st.multiselect("Select the features, sort them by importance",features_list,key="d1")
                create_hl = st.checkbox('Create D1?')
                if create_hl and include_features:
                    description_insert_partial = partial(description_insert , description = description_1)
                    search_terms_df["Description_1"] = search_terms_df.apply(description_insert_partial , axis=1)
                    search_terms_df

            create_hl_wof = st.checkbox('Create descriptio 1 without feature insertion')
            if create_hl_wof:
                search_terms_df["Description_1"] = description_1
                search_terms_df


        if st.sidebar.checkbox('Create Description 2'):
            max_text_len = st.number_input("Max Length for this description",format='%i' , value = 90 , key="d2" )
            description_2 = st.text_input("Text for description 2" , value="")
            if len(description_2) > max_text_len:
                st.warning("Text is too long")

            include_features = st.checkbox('Want To Introduce Features?',key="d2")
            if include_features:
                features_list = feature_columns.tolist()
                features_selection = st.multiselect("Select the features, sort them by importance",features_list,key="d2")
                create_hl = st.checkbox('Create D2?')
                if create_hl and include_features:
                    description_insert_partial = partial(description_insert , description = description_2)
                    search_terms_df["Description_2"] = search_terms_df.apply(description_insert_partial , axis=1)
                    search_terms_df

            create_hl_wof = st.checkbox('Create descriptio 2 without feature insertion')
            if create_hl_wof:
                search_terms_df["Description_2"] = description_2
                search_terms_df


    ########################################################### PATHS


        if st.sidebar.checkbox('Create path 1'):
            feature_select = st.selectbox("Feature for path_1"  ,options  = feature_columns )
            search_terms_df["path_1"] = search_terms_df[feature_select]

        if st.sidebar.checkbox('Create path 2'):
            feature_select = st.selectbox("Feature for path_2"  ,options  = feature_columns )
            search_terms_df["path_2"] = search_terms_df[feature_select]

        if st.sidebar.checkbox('Create final URL'):
             final_url = st.text_input("Text for final_url")
             search_terms_df["URL"] = final_url

        # search_terms_df

    ############################################################ DOWNLOAD DATA
        download_data = st.sidebar.checkbox('Download Data?' , key = "download_data")
        if download_data:
            towrite = io.BytesIO()
            search_terms_df.to_excel(towrite, sheet_name='Ads_template',index = False,  encoding = 'UTF-16')
            towrite.seek(0)  # reset pointer
            encoded = base64.b64encode(towrite.read()).decode()  # encoded object
            href = f'<a href="data:file/csv;base64,{encoded}" download ="Ads_template.xlsx">Download Excel File</a> (right-click and save as &lt;some_name&gt;.csv)'
            st.markdown(href, unsafe_allow_html=True)  

    #########----------------- AUTOBUILDER

    if module_program == "Autobuilder":

        def parallelize_test(data, func, num_of_processes=12):
            data_split = np.array_split(data, num_of_processes)
            pool = Pool(num_of_processes)
            data = pd.concat(pool.map(func, data_split))
            pool.close()
            pool.join()
            return data

        def run_on_subset_test(func, data_subset):
        #     return data_subset.apply(func)
            return func(data_subset)
        @st.cache
        def parallelize_on_rows_test(data, func, num_of_processes=12):
            return parallelize_test(data, partial(run_on_subset_test, func), num_of_processes)


        def ad_group_assignment(data_subset):

            distancias = 1 - distance.cdist(data_subset['embedding_average'].tolist()
                                                    , ad_groups_clusters_centers['embedding_average'].tolist()
                                                        , 'cosine')

            list_distances = distancias

            index_sol =  [np.where(list_distance==max(list_distance))[0][0] for list_distance in list_distances]
            del list_distances
            gc.collect()
            data_subset['test'] = np.array(index_sol)
            return data_subset['test']

        ad_groups_file = st.file_uploader("Upload Keywords and Ad groups file")
        if ad_groups_file is not None:
            ad_groups_df = pd.read_excel(ad_groups_file) 
            ad_groups_columns = ad_groups_df.columns
            for column in ad_groups_columns:
                ad_groups_df = ad_groups_df.astype({column:"str"})
                ad_groups_df[column] = ad_groups_df[column].str.lower()

            st.write("**File uploaded and read**")
            if st.checkbox('Show ad groups file'):
                st.write(ad_groups_df)

        search_terms_file = st.file_uploader("Upload Search Terms file")
        if search_terms_file is not None:
            search_terms_df = pd.read_excel(search_terms_file) 
            search_terms_df = search_terms_df
            search_terms_columns = search_terms_df.columns
            for column in search_terms_columns:
                search_terms_df = search_terms_df.astype({column:"str"})
                search_terms_df[column] = search_terms_df[column].str.lower()

            st.write("**File uploaded and read**")

            show_data_bool = st.checkbox('Show features df')

            if show_data_bool:
                st.write(search_terms_df)


        if search_terms_file is not None and ad_groups_file is not None:


            keywords_ad_groups_column = st.selectbox("Which columns has the keywords in the ad group file", options = ad_groups_columns)
            ad_groups_column = st.selectbox("Which columns has the ad groups in the ad group file", options = ad_groups_columns)
            search_term_column = st.selectbox("Which columns has the search terms in the search terms file", options = search_terms_columns)
            i1 = search_terms_df.set_index(search_term_column).index
            i2 = ad_groups_df.set_index(keywords_ad_groups_column).index
            search_terms_df = search_terms_df[~i1.isin(i2)]
        create_model_bool = st.checkbox('Create Model?')
        if create_model_bool and ad_groups_file is not None and search_terms_file is not None:
            model = create_model(ad_groups_df , column=keywords_ad_groups_column)
            st.write("Model trained")


        assign_cluster_bool = st.checkbox('Assign clusters')
        if assign_cluster_bool:

            clusters_names = ad_groups_df[ad_groups_column].unique()
            clusters_names_df = pd.DataFrame(clusters_names , columns = ['ad_group'])


            search_terms_df = create_embedding_parallel(search_terms_df , column=search_term_column)
            clusters_names_df = create_embedding_parallel(clusters_names_df , column='ad_group')
            ad_groups_clusters_centers = clusters_names_df
            search_terms_df['index'] = parallelize_on_rows_test(search_terms_df
                                                               , ad_group_assignment)
            search_terms_df['ad_group'] =  search_terms_df['index'].apply(lambda x: ad_groups_clusters_centers['ad_group'][x] )
            search_terms_df

        download_data = st.checkbox('Download Data' , key = "download_data")
        if download_data:
            towrite = io.BytesIO()
            results_to_save = search_terms_df.drop(columns=['embedding_average']) 
            results_to_save.to_excel(towrite, sheet_name='search terms',index = False,  encoding = 'UTF-16')
            towrite.seek(0)  # reset pointer
            encoded = base64.b64encode(towrite.read()).decode()  # encoded object
            href = f'<a href="data:file/csv;base64,{encoded}" download ="searches_terms_and_ad_groups.xlsx">Download Excel File</a>'
            st.markdown(href, unsafe_allow_html=True)