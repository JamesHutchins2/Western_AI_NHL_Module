#https://www.hockey-reference.com/leagues/NHL_2018_skaters.html
from collections import Counter
from responses import step1_responses, blank_spot
from user_functions import preprocess, compare_overlap, pos_tag, extract_nouns, compute_similarity
import joblib
import pandas as pd
import nltk
import tensorflow as tf
import statistics
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import seaborn as sns
from matplotlib import pyplot as plt
nltk.download('punkt',quiet=True)
nltk.download('averaged_perceptron_tagger',quiet=True)
import sys
import spacy

word2vec = spacy.load('en_core_web_sm')

exit_commands = ("quit", "goodbye", "exit",'no')

class ChatBot:


    def make_exit(self,user_message):
        for exit_command in exit_commands:
            if exit_command in user_message:
                return True
    
    def chat(self):
        print('\nHello, this is a retrieval based chat bot created by Ethan OBrien\n')
        user_message=input("""These are the current functions I am capable of:\n
        1.) I can provide you a players NHL stats given their name\n
        2.) I can predict if a player will be drafted in the NHL first round given their junior stats
        \nHow can I help you today?\n""")

        while not self.make_exit(user_message):
            user_message = self.respond(user_message)

    def find_intent_match_step1(self,step1_responses,user_message):
        bow_user_message=Counter(preprocess(user_message))
        processed_responses = [Counter(preprocess(response)) for response in step1_responses]
        similarity_list=[compare_overlap(response,bow_user_message) for response in processed_responses]
        response_index=similarity_list.index(max(similarity_list))
        return step1_responses[response_index],response_index

    def respond(self,user_message):
        best_response_step1, response_index=self.find_intent_match_step1(step1_responses,user_message)
        print(best_response_step1)
        
        
        if response_index==0:
                breaker=True
                while breaker:
                    user_message=input('Could you provide me with the name of the player (include last name in response)?\n')
                    player_stats=self.get_stats(user_message)
                    if player_stats is not None:
                        breaker=False
                        best_response=player_stats
                        
        else:
                print('I will need some various information to make that prediction\n')
                name=input("What is the player's name?\n")
                breaker=True
                while breaker:
                    G=input(f'How many goals does {name} have (integer whole number)?\n')
                    
                    if G.isdigit():
                        G=int(G)
                        breaker=False
                    else:
                        print('Please enter an integer')
                breaker=True
                while breaker:
                    A=input(f'How many assists does {name} have (integer whole number)?\n')
                    
                    if A.isdigit():
                        A=int(A)
                        breaker=False
                    else:
                        print('Please enter an integer')
                breaker=True
                while breaker:
                    GP=input(f'How many games has {name} played (integer whole number)?\n')
                    if GP.isdigit():
                        GP=int(GP)
                        breaker=False
                    else:
                        print('Please enter an integer')
                breaker=True
                while breaker:
                    PIM=input(f'What is {name} penalty minutes (integer whole number)?\n')
                    if PIM.isdigit():
                        PIM=int(PIM)
                        breaker=False
                    else:
                        print('Please enter an integer')

                breaker=True
                while breaker:
                    position_D=input(f'Is {name} Defense (only enter 1 or 0)?\n')
                    if position_D.isdigit():
                        position_D=int(position_D)
                        if position_D == 1 or position_D==0:
                            breaker=False
                    else:
                        print('Please enter an integer')

                breaker=True
                while breaker:
                    position_LW=input(f'Is {name} a Left-Winger (only enter 1 or 0)?\n')
                    if position_LW.isdigit():
                        position_LW=int(position_LW)
                        if position_LW == 1 or position_LW==0:
                            breaker=False
                    else:
                        print('Please enter an integer')
                
                

                best_response,model_list,probs_lis= self.get_predictions(name,G,A,GP,PIM,position_D,position_LW)
        
        #entity=self.find_entities(user_message)
        #stats=self.get_stats(entity)
        #print(best_response_step1.format(stats))
        print(best_response)
        print("Here is a graph with the results:\n")
        self.get_graphs(model_list,probs_lis)
        input_message = input("Do you have any other questions? \n")
        return  input_message 
                 
    def get_graphs(self,model_list,probs_lis):
        fig=plt.figure(figsize=(10,5))
        sns.barplot(x=model_list,y=probs_lis)
        plt.xlabel('Model Type')
        plt.ylabel('Probability (%)')
        ax = plt.gca()
        ax.set_ylim(0, 100)
        sns.set_context("notebook", font_scale=0.3)
        plt.show()
        plt.close()
        

    def get_predictions(self,name,G,A,GP,PIM,position_D,position_LW):
        loaded_scaler = joblib.load('scaler.joblib')
        x_titles=['G','A','GP','PIM','position_D','position_LW']
        user_input_data = pd.DataFrame(data=[[G,A,GP,PIM,position_D,position_LW]], columns=x_titles)
        scaled_user_input_data = loaded_scaler.transform(user_input_data)
        scaled_user_input_data = pd.DataFrame(data=scaled_user_input_data, columns=x_titles)
        user_input_data_single_row = scaled_user_input_data.iloc[0, :]
        user_input_data_single_row_array = user_input_data_single_row.to_numpy().reshape(1, -1)

        rf_dt_model= joblib.load('random_forest_model.joblib')
        rf_prediction = rf_dt_model.predict(user_input_data_single_row_array)[0]
        rf_prediction_prob=(rf_dt_model.predict_proba(user_input_data_single_row_array)[0][1]*100).round(2)
        print(f"The Random Forest Classifier predicts there is a {rf_prediction_prob}% probability that {name} will be drafted in the 1st round of the NHL\n")
        
        bag_svm_model= joblib.load('bag_svm_model.joblib')
        svm_prediction = bag_svm_model.predict(user_input_data_single_row_array)[0]
        svm_prediction_prob=(bag_svm_model.predict_proba(user_input_data_single_row_array)[0][1]*100).round(2)
        print(f"The Support Vector Machine Classifier with Bagging predicts there is a {svm_prediction_prob}% probability that {name} will be drafted in the 1st round of the NHL\n")

        bag_dtree_model= joblib.load('bag_dtree_model.joblib')
        dtree_prediction = bag_dtree_model.predict(user_input_data_single_row_array)[0]
        dtree_prediction_prob=(bag_dtree_model.predict_proba(user_input_data_single_row_array)[0][1]*100).round(2)
        print(f"The Decision Tree Classifier with Bagging predicts there is a {dtree_prediction_prob}% probability that {name} will be drafted in the 1st round of the NHL\n")


        bag_lr_model= joblib.load('bag_lr_model.joblib')
        lr_prediction = bag_lr_model.predict(user_input_data_single_row_array)[0]
        lr_prediction_prob=(bag_lr_model.predict_proba(user_input_data_single_row_array)[0][1]*100).round(2)
        print(f"The Logistic Regression Classifier with Bagging predicts there is a {lr_prediction_prob}% probability that {name} will be drafted in the 1st round of the NHL\n")


        simple_MLP_model= tf.keras.models.load_model('simple_MLP_model.h5')
        sim_mlp_prediction=simple_MLP_model.predict(user_input_data_single_row_array,verbose=False).round()[0][0]
        sim_mlp_prediction_prob=(simple_MLP_model.predict(user_input_data_single_row_array,verbose=False)[0][0]*100).round(2)
        print(f"The Simple MLP Deep Learning Classifier predicts there is a {sim_mlp_prediction_prob}% probability that {name} will be drafted in the 1st round of the NHL\n")

        complex_MLP_model= tf.keras.models.load_model('complex_MLP_model.h5')
        com_mlp_prediction=complex_MLP_model.predict(user_input_data_single_row_array,verbose=False).round()[0][0]
        com_mlp_prediction_prob=(complex_MLP_model.predict(user_input_data_single_row_array,verbose=False)[0][0]*100).round(2)
        print(f"The Complex MLP Deep Learning Classifier predicts there is a {com_mlp_prediction_prob}% probability that {name} will be drafted in the 1st round of the NHL\n")

        prediction_list=[rf_prediction,svm_prediction,dtree_prediction,lr_prediction,sim_mlp_prediction,com_mlp_prediction]
        models_list=['Random Forest','Support Vector Machine','Decision Tree','Logistic Regression','Simple MLP Deep Learning','Complex MLP Deep Learning']
        probs_list=[rf_prediction_prob,svm_prediction_prob,dtree_prediction_prob,lr_prediction_prob,sim_mlp_prediction_prob,com_mlp_prediction_prob]

        prediction=statistics.mode(prediction_list)
        avg_prob=np.mean(probs_list).round(2)

        if prediction==1:
            prediction_message=f"Overall, we predict that there is a {avg_prob}% probability that {name} will be drafted in the first round."
        else: 
            prediction_message=f"Overall, we predict that there is a {100-avg_prob}% probability that {name} will not be drafted in the first round."

        return prediction_message,models_list,probs_list


    def get_stats(self,user_message):
        user_message=preprocess(user_message)
        player_names=pd.read_csv('player_names.csv')
        player_names=player_names.Player.to_list()
        for word in user_message:
            for player in player_names:
                if word in player.split()[1]:
                    entity=player

        if entity==None:
            print("I'm sorry, I could not find that player. Could you please try again?\n") 
            return None 
        breaker=True
        df=pd.read_csv('stats.csv')
        df.year=df.year.apply(lambda x: str(x))
        years=list(df.year.unique())
        while breaker: 
            user_message=input(f'For which year would you like NHL stats for {entity} (2017-2020 season data available)?\n')
            user_message=str(user_message)
            user_message=preprocess(user_message)
            for word in user_message:  
                if word in years:
                    stats=df[df['year'].str.contains(word,case=False)]
                    player_stats=stats[stats['Player'].str.contains(entity,case=False)]
                    player_stats.set_index('Player', inplace=True)
                    player_stats.drop(columns=['Unnamed: 0'], inplace=True)
                    pd.options.display.max_columns = None
                    pd.options.display.width = None
                    pd.options.display.max_colwidth = None
                    pd.options.display.precision = 2
                    breaker=False
                    return player_stats


    def find_entities(self,user_message):
        tagged_user_message=pos_tag(preprocess(user_message))
        message_nouns=extract_nouns(tagged_user_message)
        tokens=word2vec(' '.join(message_nouns))
        category=word2vec(blank_spot)
        word2vec_result=compute_similarity(tokens,category)
        word2vec_result.sort(key=lambda x: x[2])
        if len(word2vec_result) < 1:
          return blank_spot
        else:
         return word2vec_result[-1][0]



hockey_bot = ChatBot()
hockey_bot.chat()