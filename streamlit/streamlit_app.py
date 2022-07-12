import streamlit as st
from streamlit_shap import st_shap

import pandas as pd
import numpy as np
import io
import base64
import pickle
import sklearn
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import shap
shap.initjs() # for visualization

########################################################################################################################################################################
# Définition du main()
########################################################################################################################################################################

def main():
	st.sidebar.title("RainsBerry")
	#st.set_page_config(
	#page_title="RainsBerry - Météo",
	#page_icon="👋",
	#layout="wide",)
	Menu = st.sidebar.radio(
		"Menu",
		('Le Projet Météo', 'Dataset & PreProcessing','DataViz','Modelisations','Performances','Simulations','Clustering','Séries Temporelles','Conclusion'))
	if Menu == 'Le Projet Météo':
		from PIL import Image
		image = Image.open('images/RainsBerry_2.jpg')
		st.image(image,width=600,caption="")
		'''
		* Le projet présenté dans ce streamlit a été développé dans le cadre de la formation Data Scientist de Datascientest.com - Promotion Octobre 2021.
		* L'objectif premier de ce projet est de mettre en application les différents acquis de la formation sur la problématique de prévision météo et plus précisément de répondre à une question essentielle: va-t-il pleuvoir demain?
		'''
		st.image('images/Intro_météo.jpg',width=650,caption="")
		'''
		* En dehors d'intéresser particulièrement les fabricants de parapluie, on comprend aussi que cette question est essentielle que ce soit dans le domaine des loisirs (gestion des parcs d'attraction), de l'agriculture, du traffic routier, et bien d'autres sujets.
		* Le lien du repo github est disponible ici: https://github.com/DataScientest-Studio/RainsBerryPy.
		'''
	if Menu == 'Dataset & PreProcessing':
		PreProcessing()
	if Menu == 'DataViz':
		DataViz()
	if Menu == 'Modelisations':
		Modelisations()
	if Menu == 'Performances':
		Performances()
	if Menu == 'Simulations':
		simulation()
	if Menu == 'Clustering':
		clustering()
	if Menu == 'Conclusion':
		conclusion()
	if Menu == 'Séries Temporelles':
		serie_temp()
	if Menu == 'Rapport':
		rapport()
	st.sidebar.text("")
	st.sidebar.text("Projet DataScientest")
	st.sidebar.text("Promotion DataScientist Octobre 2021")
	st.sidebar.text("Lionel Bottan")
	st.sidebar.text("Julien Coquard")
	st.sidebar.text("Samuel Guérin")
	st.sidebar.write("[Lien du git](https://github.com/DataScientest-Studio/RainsBerryPy)")

########################################################################################################################################################################
# Définition de la partie Preprocessing
########################################################################################################################################################################
    
def PreProcessing():
	from PIL import Image
	st.header("Dataset & PreProcessing")
	image = Image.open('images/weatherAUS.jpg')
	st.image(image, caption='Relevé Météo en Australie',width=600)
	st.subheader("Dataset originel")
	df=pd.read_csv('data/weatherAUS.csv') #Read our data dataset
	buffer = io.StringIO()
	df.info(buf=buffer)
	s = buffer.getvalue()
	st.write("Présentation du jeu de données : ")
	'''
	Le jeu de données possède 145 460 entrées et 23 colonnes dont :
	* La date de l'observation (Date).
	* La ville dans laquelle se situe la station météo (Location). 
	* La variable cible RainTomorrow dont la valeur (Yes ou No) indique s'il a plu ou non le lendemain de l'observation.
	* 20 variables décrivant les conditions atmosphériques du jour de l’observation :
	'''
	st.text(s)
	'''
	Remarques :
	* Les valeurs de la variable RainToday (Yes, No) sont définies par la variable Rainfall (Yes si précipitations > 1mm).
	* Plusieurs variables possèdent de nombreuses valeurs manquantes que l'on a géré de la manière suivante:
	* Soit par exclusion pur et simple des entrées avec valeurs manquantes
	* Soit par l'utilisation d'un transformeur KNN: https://medium.com/@kyawsawhtoon/a-guide-to-knn-imputation-95e2dc496e
	'''
	st.subheader("Ajout de nouvelles données")
	st.write("Principaux climats australiens",width=600)
	image = Image.open('images/grd_climats.png')
	st.image(image, caption='Climats australiens',width=600)
	st.write("Classification de Köppen")
	image = Image.open('images/clim_koppen.png')
	st.image(image, caption='Climats - Classification de Koppen',width=600)
	df=pd.read_csv('data/climatsAUS_v2.csv') #Read our data dataset
	buffer = io.StringIO()
	df.info(buf=buffer)
	s = buffer.getvalue()
	st.write("Présentation du jeu de données : ")
	st.text(s)
	st.write("Coordonnées GPS")
	image = Image.open('images/GPS.jfif')
	st.image(image, caption='Coordonnées GPS')
	df=pd.read_csv('data/aus_town_gps.csv') #Read our data dataset
	buffer = io.StringIO()
	df.info(buf=buffer)
	s = buffer.getvalue()
	st.write("Présentation du jeu de données : ")
	st.text(s)
	'''
	###Preprocessing
	Création de nouvelles données:
	* Numérisation des deux variables booléennes RainToday et RainTomorrow.
	* Décomposition de la date en trois variables : Année, Mois, Jour.
	* Climat_Koppen : classe climatique dans la classification de Köppen.
	* Clim_type : type de climat regroupant plusieurs classes de Köppen, définie à partir de Climat_Koppen
	* Ajout de 3x3 variables précisant la direction des vents (définies à partir de WindGustDir, WindDir9am et WindDir3pm) :
		* WindGust_Ang, Wind9am_Ang, Wind3pm_Ang : angle correspondant (en degrés) sur le cercle trigonométrique (ie. E=0° et rotation dans le sens direct).
		* WindGust_cos, Wind9am_cos, Wind3pm_cos : cosinus de l'angle (abscisse des coordonnées trigo).
		* WindGust_sin, Wind9am_sin, Wind3pm_sin : sinus de l'angle (ordonnée des coordonnées trigo). 
	* Pluie à J-1, J-2, J+1, J+2
	* Circularisation de la variable Mois (https://datascientest.com/numeriser-des-variables). De cette façon, les mois de décembre et janvier ont des valeurs proches.
	###
	### Gestion des valeurs manquantes:
	* DropNa pour Manière brute: 56k entrées
	* Interpolate et KNN imputer: 145k entrées
	'''
	


########################################################################################################################################################################
# Définition de la partie DataViz
########################################################################################################################################################################
 
def DataViz():
    st.header("DataViz")
    if st.checkbox("Corrélations de la pluie du lendemain (RainTomorrow) et de  l'ensoleillement (Sunshine)"):
        st.image('images/Dataviz_corr.jpg')
        '''
        #### Observations :
        * L’analyse des corrélations nous montre que les liaisons entre les différents critères sont nombreuses.
        * Quelles sont les variables les plus corrélées à RainTomorrow ?
            * Ensoleillement : Sunshine
            * Humidité : 3pm et 9am
            * Couverture nuageuse : 3pm et 9am
            * Pluie du jour : RainToday
            * Pression atmosphérique : Pressure3pm et Pressure9am
        * L'ensoleillement (Sunshine) est corrélé à RainTomorow_num malgré presque 50% de valeurs manquantes pour cette variable. Quand on regarde les corrélations, on peut imaginer de traiter ces valeurs manquantes en régressant Sunshine sur les critères les plus corrélés, à savoir :
            * Couverture nuageuse : 3pm et 9am
            * Humidité : 3pm et 9am
            * Température : Temp3pm, MaxTemp, Temp9am
        '''       
    if st.checkbox("Cartographie"):
        st.image('images/Dataviz_carto.jpg')
        '''
        #### Observations : 
        * Les stations météo d'Australie sont regroupées en 4 climats différents :
            * méditerrannéen : stations du sud-ouest et du sud-centre
            * chaud_humide (tropical et subtropical humide) => côte est du pays
            * tempéré_froid (tempéré océanique + montagnard) => plutôt sud-est
            * sec (chaud et semi-aride, voire aride) => intérieur du pays
        * La distribution mensuelle des précipitations illustre bien les différences de climat (mousson estivale pour le climat tropical, hivernale pour le climat méditerranéen).
        * Pour les stations au climat sec, on observe 9% de jours de pluie alors que pour les autres on est aux alentours de 22, 23%.
        '''       
    if st.checkbox("Influence sur la pluie du lendemain"):
        st.image('images/Dataviz_influence.jpg')
        '''
        #### Constats :
        * La distribution des variables Sunshine et Humidity3pm est bien différente selon RainTomorrow.
        * Pour MinTemp, la distribution est relativement similaire.
        * Pour Rainfall et Evaporation, il faut appliquer la fonction log pour neutraliser l'influence des valeurs extrêmes. On voit aussi l'influence plus importante de Rainfall sur RainTomorrow (distribution différente).
        '''      
	
########################################################################################################################################################################
# Définition de la partie modélisation
########################################################################################################################################################################

def Modelisations():
    st.header("Modélisations")
    
    Menu_mod = st.sidebar.radio(
     "Menu Modélisations",
     ('Equilibrage des classes','Traitement des valeurs manquantes','Sélection de variables', 'Conclusion'))

    def Equilibrage():
        st.subheader("Équilibrage des classes")
        st.image('images/model_01_desequilibre.jpg')
        st.markdown("**Performances d'un modèle Random Forest sur le jeu de données complet :**")
        st.image('images/model_02_sans_equ.jpg')
        if st.checkbox("Après équilibrage"):
            st.image('images/model_03_avec_equ.jpg')
            st.image('images/model_04_PrecRap.jpg')
        if st.checkbox("Modification du seuil de détection de la classe 1"):
            st.image('images/model_05_seuils_proba.jpg')
            st.image('images/model_06_seuilmaxF1.jpg')
	            
    def TraitementNA():
        st.subheader("Traitement des valeurs manquantes")
        '''
        #### **Hypothèse : Les performances dépendent de la méthode de traitement des valeurs manquantes.**
        ##### Trois techniques ont été utilisées pour traiter les valeurs manquantes et créer trois jeux de données : 
        * Remplacement des valeurs manquantes par la méthode KNN-Imputer.
        * Suppression des observations possédant des valeurs manquantes par la méthode dropna.
        * Suppression des quatre variables possédant le plus de valeurs manquantes, puis suppression des observations restantes possédant des NaN.
        '''       
        st.image('images/model_07_proportionsNA.jpg')
        if st.checkbox("Scores"):
            st.markdown("**Scores en fonction du jeu de données :**")
            st.image('images/model_08_scores_JD.jpg')
            '''
            ##### Conclusion : Le jeu de données dropna présentent les meilleures performances, en plus d'être le plus rapide.
            '''
	
    def SelectionVar():
        st.subheader("Sélection de variables")
        '''
        #### **Hypothèse : Des variables peu pertinentes perturbent le modèle, ce qui affecte ses performances.**
        '''
        st.image('images/model_09_selectKBest.jpg') 
        '''
        ##### Conclusion : À partir de six variables, on observe une croissance de toutes les métriques au fur et à mesure qu’on intègre des variables au modèle. 
        ##### Il n'est donc pas nécessaire de supprimer des variables pour améliorer les scores.
        '''
  
    def Conclusion():
        st.subheader("Conclusion")
        '''
        * Le rééchantillonnage permet d'obtenir des scores légèrement meilleurs, mais c'est surtout le choix du seuil de décision qui a le plus d'impact sur les performances.
        * L'interpolation des valeurs manquantes par KNN Imputer réduit les performances au lieu de les améliorer. Il est préférable d’utiliser un jeu de données où les valeurs manquantes ont simplement été supprimées.
        * Le retrait de certaines variables n’améliore pas les performances. Toutes les variables peuvent être utilisées pour entrainer nos modèles.
        '''
         
    if Menu_mod == 'Equilibrage des classes':
        Equilibrage()
        
    if Menu_mod == 'Traitement des valeurs manquantes':
        TraitementNA()
        
    if Menu_mod == 'Sélection de variables':
        SelectionVar()
        
    if Menu_mod == 'Conclusion':
        Conclusion()

########################################################################################################################################################################
# Définition de la partie perfomance
########################################################################################################################################################################

def Performances():
    st.header("Performances des modèles testés")
    '''
    #### Les algorithmes suivants ont été testés en prenant en compte les résultats des analyses précédentes :
    * Rééquilibrage du jeu de données avec RandomUnderSampler. 
    * Conservation de toutes les variables prédictives.
    * Choix de l'algorithme sur le dataset sans les NA (données réelles)
    * En revanche, application possible sur les données interpolées ce qui aurait l'intérêt de pouvoir avoir des prédictions sur les observations qui ont des valeurs manquantes (par exemple, les stations  qui ne mesurent pas certains indicateurs). 

    #### Liste des algorithmes testés :
    * Arbre de décision
    * Boosting sur arbre de décision (Adaboost classifier)
    * Isolation Forest (détection d’anomalies) => non présenté car vraiment trop dégradé.
    * Régression logistique
    * SVM
    * KNN
    * Random Forest
    * Light GBM
    * Bagging Classifier
    * Stacking Classifier (avec les modèles préentrainés RandomForest, SVM et LogisticRegression)
	
    ##### Optimisation des modèles :
    * Une grille de recherche sur les hyperparamètres a été construite pour les modèles avec le choix de maximiser le f1 comme métrique de performance et 3 folds pour limiter le surapprentissage.

    ##### Choix du modèle :
    * Le modèle final sera choisi au regard de la courbe de ROC, de l'AUC globale et surtout des métriques f1_score, precision, rappel sur la classe à modéliser.

    ##### Définitions :
    * La precision correspond au taux de prédictions correctes parmi les prédictions positives. Elle mesure la capacité du modèle à ne pas faire d’erreur lors d’une prédiction positive.
    * Le recall correspond au taux d’individus positifs détectés par le modèle. Il mesure la capacité du modèle à détecter l’ensemble des individus positifs.
    * Le F1-score évalue la capacité d’un modèle de classification à prédire efficacement les individus positifs, en faisant un compromis entre la precision et le recall (moyenne harmonique).
    ''' 
    if st.checkbox("Courbe de ROC"):
        st.image('images/Perf_ROC.jpg')       
    if st.checkbox("Selon le seuil de détection"):
        st.image('images/Perf_seuils.jpg')
        st.image('images/Perf_seuils1.jpg')
    if st.checkbox("Deep Learning"):
        '''
        L’objectif de cette section est de tester des modèles de Deep Learning pour prédire RainTomorrow et de comparer les performances obtenues aux modèles de Machine Learning classique présentés ci dessus.
        * 2 types de réseaux de neurones ont été testés :
        '''
        if st.checkbox("1-Modèles denses classiques"):
            '''
            * Plusieurs modèles ont été construits en faisant varier les caractéristiques suivantes :
                * Augmentation du nombres de couches : de 4 à 5 couches de neurones
                * Augmentation du nombre de neurones des couches
                * Changement de la fonction d’activation : tanh, ReLu
                * Changement de l’initialisateur : normal, Xavier, HeNormal
                * Diminution de la taille du batch : 32, 16
            * L’ensemble des résultats obtenus par les premiers modèles sont assez similaires, les performances ne sont pas significativement différentes, en particulier si l’on considère les variations d’un entrainement à l’autre. 
            * Le meilleur modèle donne les résultats suivants (après rééchantillonnage) :
            '''
            st.image('images/DeepLearning_dense.jpg')
        if st.checkbox("2-Fast AI"):
            '''
            * Pour compléter l’étude ci-dessus, un modèle de Deep Learning utilisant la bibliothèque FastAI a été développé en s’inspirant de la littérature disponible sur le web :
            * Les performances obtenues par le modèle sont reportées dans les tableaux ci-dessous :
            '''
            st.image('images/DeepLearning_FastAI.jpg')
        if st.checkbox("3-Conclusion Deep Lerning"):
            '''
            * Les modèles de Deep Learning développés n’ont pas démontré de meilleurs résultats que les modèles de Machine Learning classique étudiés en début de projet.
            * Par ailleurs, au-delà des performances peu convaincantes sur notre jeu de données, le manque d’interprétabilité des modèles de Deep Learning par rapport au Machine Learning classique ne pousse pas à les développer davantage lors de ce projet.
            '''        
    if st.checkbox("Conclusion"):
        '''
        * La comparaison des algorithmes sur la courbe de ROC nous donne une liste de quatre algorithmes sensiblement plus performants que les autres :
            * la Random Forest
            * le Bagging
            * la XGBoost
            * la Light GBM
        
        * Les comparaisons sur le F1_score en choisissant différents seuils de probabilités (0.50, F1_max, recall=precision) vont nous conduite à préférer la XGBOOST qui est légèrement plus performante que la lightGBM sur le seuil "recall=precision".
        * Les modèles de Deep Learning développés n’ont pas démontré de meilleurs résultats que les modèles de Machine Learning classique étudiés en début de projet.
        '''
        st.image('images/Perf_conclusion1.jpg')
        if st.checkbox("Interprétabilité de notre modèle final XGBOOST"):
            '''
            * L'interprétabilité est importante dès que les résultats d'un modèle influent grandement sur des décisions importantes. En entreprise par exemple, expliquer à des équipes non-initiées le fonctionnement d'un modèle pose toujours son lot de défis.
            * Ici nous ne présentons que l'interprétabilité  avec Shapash (Shapash est une librairie Python qui vise à rendre le Machine Learning intelligible par le plus grand nombre. Concrètement, il s’agit d’une surcouche à d’autres librairies d’intelligibilité (Shap, Lime))
            * Interprétabilité globale
            '''
            st.image('images/Interpretabilite_globale.jpg')
            '''
            * Interprétabilité locale
            '''            
            st.image('images/Interpretabilite_locale.jpg')

        
########################################################################################################################################################################
# Définition de la partie simulation
########################################################################################################################################################################

def simulation():
    #Chargement du modele
    picklefile = open("modeles/xgboost.pkl", "rb")
    modele = pickle.load(picklefile)  

    #Definition des features
    features = ["RainToday_Num","Rain_J-1","Rain_J-2","MinTemp","MaxTemp","Sunshine","Evaporation",
        "Humidity3pm","Humidity9am","Pressure9am","Pressure3pm","Cloud3pm","Cloud9am", 
        "Wind9am_cos","Wind3pm_cos","WindGust_cos","Wind9am_sin","Wind3pm_sin","WindGust_sin", 
        "Mois","Clim_type_det"]
                
    st.markdown("# Simulation")

    st.subheader("Lecture des données")

    Data = st.selectbox("DataFrame: " , ["echantillon","Sydney","AliceSprings","Darwin","Perth","Hobart"])

    if ( Data == "echantillon"):
        df=pd.read_csv('data/echantillon.csv') #Read our data dataset
    if ( Data == "Sydney"):
        df=pd.read_csv('data/Sydney.csv') #Read our data dataset
    if ( Data == "AliceSprings"):
        df=pd.read_csv('data/AliceSprings.csv') #Read our data dataset
    if ( Data == "Darwin"):
        df=pd.read_csv('data/Darwin.csv') #Read our data dataset
    if ( Data == "Perth"):
        df=pd.read_csv('data/Perth.csv') #Read our data dataset
    if ( Data == "Hobart"):
        df=pd.read_csv('data/Hobart.csv') #Read our data dataset    

    st.write("Nombre de lignes : ", df.shape[0]) 
    st.write("Nombre de colonnes : ", df.shape[1]) 

    st.subheader("DataViz")

    DataViz = st.selectbox("Quelle Dataviz ? : " , ["Part jours de Pluie","Correlation","Analyse mensuelle","Impact de RainTomorrow"])

    if ( DataViz == "Part jours de Pluie"):
        #Part des jours de pluie
        fig = plt.figure(figsize=(3,3))
        x = df.RainTomorrow_Num.value_counts(normalize=True)
        colors = sns.color_palette('pastel')[0:5]
        labels = ['Pas de pluie', 'Pluie']
        plt.pie(x, labels = labels, colors = colors, autopct='%.0f%%')
        plt.title("Part des jours de pluie")
        st.write(fig)

    if ( DataViz == "Correlation"):
        fig, ax = plt.subplots(figsize=(15,6))
        ListeCrit = ["RainTomorrow_Num","MinTemp","MaxTemp","Sunshine","Evaporation","Humidity3pm"]
        sns.heatmap(df[ListeCrit].corr(), cmap="YlGnBu",annot=True,ax=ax)
        st.write(fig)

        fig = plt.figure( figsize= (20, 7) )
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        corr = df.corr()
        ax1.title.set_text('Correlations de RainTomorrow')
        temp = corr[["RainTomorrow_Num"]].loc[abs(corr["RainTomorrow_Num"]) > 0.2].sort_values(by="RainTomorrow_Num",ascending=False)
        sns.heatmap(temp, cmap="YlGnBu",annot=True,ax=ax1)
        ax2.title.set_text('Correlations de Sunshine')
        temp = corr[["Sunshine"]].loc[abs(corr["Sunshine"]) > 0.2].sort_values(by="Sunshine",ascending=False)
        sns.heatmap(temp , cmap="YlGnBu",annot=True,ax=ax2)
        st.write(fig)


    if ( DataViz == "Analyse mensuelle"):
        fig, ax = plt.subplots(figsize=(15,6))
        ax.title.set_text("Distribution mensuelle des pluies")
        sns.lineplot(ax=ax,data=df, x="Mois", y="Rainfall")
        st.write(fig)

    if ( DataViz == "Impact de RainTomorrow"):
        fig, ax = plt.subplots(figsize=(20,4))
        plt.subplot(131)
        sns.histplot(data=df, x="Sunshine",hue="RainTomorrow_Num",bins=20, multiple="layer", thresh=None)
        plt.subplot(132)
        sns.histplot(data=df, x="MinTemp",hue="RainTomorrow_Num",bins=20, thresh=None)
        plt.subplot(133)
        sns.histplot(data=df, x="Humidity3pm",hue="RainTomorrow_Num",bins=20)
        st.write(fig)

    st.subheader("Prédiction")

    if st.button("Predict"):  
        #Courbe de ROC
        probs = modele.predict_proba(df[features])
        y_test =  df["RainTomorrow_Num"]
        fpr, tpr, seuils = sklearn.metrics.roc_curve(y_test, probs[:,1], pos_label=1)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        fig = plt.figure(figsize=(15,6))
        plt.plot(fpr, tpr, color='purple',  linestyle='--', lw=1, label='Model (auc = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle=':', label='Aléatoire (auc = 0.5)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux faux positifs')
        plt.ylabel('Taux vrais positifs')
        plt.title('Courbe ROC')
        plt.legend(loc="lower right");
        st.pyplot(fig)
    
        #Graphe selon le seuil 
        precision, recall, thresholds = precision_recall_curve(y_test, probs[:, 1], pos_label=1)
        dfpr = pd.DataFrame(dict(precision=precision, recall=recall, threshold=[0] + list(thresholds)))
        dfpr['F1']= 2 * (dfpr.precision * dfpr.recall) / (dfpr.precision + dfpr.recall)
        dfrpr_maxF1 = dfpr[dfpr.F1 == dfpr.F1.max()].reset_index()
        Seuil = dfrpr_maxF1["threshold"].values[0]
        dfpr["Diff_Recall_Precision"] = np.abs(dfpr["recall"]-dfpr["precision"])
        dfrpr_MinDiff = dfpr[dfpr.Diff_Recall_Precision == dfpr.Diff_Recall_Precision.min()].reset_index()
        Seuil1 = dfrpr_MinDiff["threshold"].values[0]
    
        fig = plt.figure(figsize=(15,6))
        plt.plot(dfpr["threshold"], dfpr['precision'],label="precision")
        plt.plot(dfpr["threshold"], dfpr['recall'],label="recall")
        plt.plot(dfpr["threshold"], dfpr['F1'],label="F1")
        plt.axvline(x=0.50,color="gray",label="seuil à 0.50")
        plt.axvline(x=Seuil,color="red",label="seuil maximisant F1")
        plt.axvline(x=Seuil1,color="purple",label="seuil Recall=Precision")
        plt.title("Choix du seuil sur la classe à modéliser")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        st.pyplot(fig)
        #Matrice de confusion
        y_pred = np.where(probs[:,1] >= 0.50, 1, 0)    
        y_pred_best = np.where( probs[:,1] >= Seuil, 1, 0)
        y_pred_best1 = np.where( probs[:,1] >= Seuil1, 1, 0)
        st.text('Matrice de confusion seuil 0.50 :\n ' + classification_report(y_test, y_pred))
        st.text('Matrice de confusion seuil maximisant F1 :\n ' + classification_report(y_test, y_pred_best))
        st.text('Matrice de confusion seuil Recall=Precision :\n ' + classification_report(y_test, y_pred_best1))    
        fig = plt.figure(figsize=(15,6))
        cm = confusion_matrix(y_test, y_pred_best)
        ConfusionMatrixDisplay(cm).plot()
        st.pyplot(fig)
        #Predictions
        prediction = modele.predict(df[features])
        predDf = pd.DataFrame(prediction,columns=["prediction"])
        Sortie = pd.concat([df[["Date","Location","Climat_Koppen","Clim_type_det","RainTomorrow_Num"]],predDf],axis=1)
        #st.write(Sortie)

    #st.subheader("Interprétabilité")
    
    #if st.button("Importance des features"):
    #    picklefile = open("modeles/xgboost.pkl", "rb")
    #    modele = pickle.load(picklefile)  
    #    explainer = shap.TreeExplainer(modele)
    #    shap_values = explainer.shap_values(df[features])
    #    st_shap(shap.summary_plot(shap_values, df[features]),height=300)

########################################################################################################################################################################
# Définition de la partie séries temporelles
########################################################################################################################################################################
    
def serie_temp():
	Menu_mod = st.sidebar.radio("Séries temporelles",('Introduction & méthodologie','1ère étude','2nde étude'))	
		
	def Intro():
		st.subheader('')
		'''
		## Introduction & Méthodologie
		Cette section traite des séries temporelles sur différents indicateurs. Notre choix s’est porté sur les indicateurs suivants :
		* RainFall, le niveau de précipitation en mm
		* Humidity3pm, le taux d'humidité à 15h
		* MaxTemp, la température maximale.
		Deux études distinctes ont été menées :
		* Étude sur sept villes représentatives des climats australiens:
		'''
		st.image('images/ST_Liste_Villes.jpg',width=600)
		st.image('images/ST_Carte_Villes.jpg',width=600)
		'''
		* Étude sur deux climats aux saisons des pluies opposées, en regroupant l’ensemble des stations. Cette étude se limitera à Rainfall:
		'''
		st.image('images/ST_Carte_Villes_2.jpg',width=600)
		'''
		## Méthodologie:
		* Interpolation des valeurs manquantes sur les données quotidiennes.
		* Prévisions faites sur les données mensuelles.
		* Conservation des 24 derniers mois comme base de validation des modèles.
		* Algorithmes testés :
			* Autoarima : pour trouver les meilleurs paramètres des SARIMA
			* SARIMAX : pour appliquer notre modèle final (qui peut être ajusté par rapport à l’Autoarima)
			* Prophet (algorithme de Facebook) en complément de SARIMAX.
		* Comparaison des performances des modèles : 
			* Deux métriques de mesure de l’erreur :
				* RMSE (erreur moyenne quadratique) : MaxTemp et Humidity3pm
				* WMAPE (Weighted Mean Absolute Percentage Error) pour RainFall
				=> Métrique intéressante pour évaluer les erreurs lorsque les valeurs réelles sont nulles ou proches de zéro. (https://resdntalien.github.io/blog/wmape/)
				* Pourcentage de corrélation de Pearson entre les valeurs réelles et prédites.
		Remarque : D’autres métriques, telle que la MAE, ont été calculées. Elles présentent toutes des résultats concordants pour l’ensemble des modèles testés et ne seront pas présentées.
		'''
		
	def Results_1():
		st.subheader('')
		'''
		# Étude sur les villes représentatives des climats australiens -
		## Visualisation de l’évolution des moyennes mensuelles pour les trois indicateurs:
		'''
		st.image('images/ST_CourbeIndic_Rainfall.jpg',width=600)
		st.image('images/ST_CourbeIndic_Hum3pm.jpg',width=600)
		st.image('images/ST_CourbeIndic_MaxTemp.jpg',width=600)
		'''
		## Observations et interprétations:
		* La saisonnalité de Rainfall est particulièrement marquée pour Cairns et Darwin avec un pic de précipitations important en février. Ces deux villes étant situées en climat tropical, elles possèdent une période de mousson importante en été.
		* Pour Humidity3pm, la saisonnalité n’est pas très marquée mais les niveaux sont bien différents entre AliceSprings (climat sec) et Norfolk Island (climat humide).
		* MaxTemp possède une saisonnalité importante pour les villes situées au sud (climats méditerranéen et océanique), tandis que les villes situées plus proche de l’équateur (Cairns et Darwin – climat tropical) présentent un hiver beaucoup plus doux et donc une saisonnalité moins marquée.
		Les deux sous-sections suivantes détaillent les résultats obtenus pour deux villes : Canberra et Cairns.
		## Résultats obtenus pour :
		'''
		if st.checkbox('Canberra'):
			st.image('images/ST_ResultTab_Camberra.jpg',width=600)
			st.image('images/ST_ResultCurv_Camberra_Rainfall.jpg',width=600)
			st.image('images/ST_ResultCurv_Camberra_Hum3pm.jpg',width=600)
			st.image('images/ST_ResultCurv_Camberra_MaxTemp.jpg',width=600)
		if st.checkbox('Cairns'):
			st.image('images/ST_ResultTab_Cairns.jpg',width=600)
			st.image('images/ST_ResultCurv_Cairns_Rainfall.jpg',width=600)
			st.image('images/ST_ResultCurv_Cairns_Hum3pm.jpg',width=600)
			st.image('images/ST_ResultCurv_Cainrs_MaxTemp.jpg',width=600)
		'''
		## Conclusion:
		Comme on pouvait s’y attendre, les variations aléatoires quotidiennes rendent les prédictions plus difficiles sur Rainfall que sur MaxTemp, comme le montre la superposition des courbes des prédictions et de la série originelle. Pour MaxTemp, le coefficient de corrélation dépasse en effet 90 % pour tous les modèles. Humidity3pm présente sur ce point, un profil intermédiaire. Pour les trois indicateurs météorologiques, les performances sont meilleures sur Cairns que sur Canberra. La différence entre les deux villes est particulièrement marquée pour Rainfall, avec un coefficient de corrélation de 61 % pour Cairns (comparable à celui d’Humidity), alors qu’il n’est que de 20 % pour Canberra. Cette différence peut s’expliquer si l’on prend en compte le climat des deux villes. Cairns présente en effet un climat tropical, avec des saisons plus marquées en termes de précipitations que Canberra, dont le climat est océanique.
		'''
		
	def Results_2():
		st.subheader('')
		'''
		# Étude sur deux climats aux saisons des pluies opposées -
		## Ici on s’intéresse non plus à une ville mais à la moyenne mensuelle de l’ensemble des villes d’un climat donné.
		## Hypothèse de travail : 
		* La variable Rainfall présente une forte périodicité pour les climats caractérisés par une période de mousson : 
		* climat tropical (Aw + Am)
		* et climat méditerranéen (Csa + Csb).
		La période de mousson est différente en climat méditerranéen (mousson hivernale) et en climat tropical (mousson estivale). Il est donc nécessaire d'étudier ces deux climats séparément.
		La méthodologie est la même que celle utilisée pour les analyse par ville.
		## Observations:
		'''
		st.image('images/ST_CourbeIndic_Rainfall_climat.jpg',width=600)
		st.image('images/ST_CourbeIndic_Rainfall_climat_saison.jpg',width=600)
		'''
		Les graphiques confirment notre hypothèse : les deux séries possèdent une forte saisonnalité mais avec un décalage d'une demi-période environ.
		La moyenne mobile, calculée sur 12 mois, évolue peu, mais les séries ne sont pas complètement stationnaires. 
		Le climat tropical présente notamment une diminution des pics de précipitations après 2012.
		'''
		st.image('images/ST_ResultTab_RainfallClimat.jpg',width=600)
		st.image('images/ST_ResultCurv_Rainfall_med.jpg',width=600)
		st.image('images/ST_ResultCurv_Rainfall_trop.jpg',width=600)
		'''
		## Conclusion
		Les performances sont meilleures sur ces deux climats que sur les villes prises indépendamment, avec des erreurs plus faible et un coefficient de corrélation dépassant les 75 %. 
		On remarque aussi de performances légèrement meilleures pour le climat méditerranéen que pour le climat tropical, si l’on considère l’erreur WMAPE. Cette différence peut s’interpréter par une meilleure stationnarité de la série méditerranéenne, visible en observant la courbe de la moyenne mobile
		'''
		
	if Menu_mod == 'Introduction & méthodologie':
		Intro()
	if Menu_mod == '1ère étude':
		Results_1()
	if Menu_mod == '2nde étude':
		Results_2()

	

########################################################################################################################################################################
# Définition de la partie rapport
########################################################################################################################################################################
  
  
def rapport():
    st.write("[Lien git_hut :](https://github.com/DataScientest-Studio/RainsBerryPy)")
    def show_pdf(file_path):
        with open(file_path,"rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

    show_pdf('https://github.com/SamuelGuerin-Git/RainsBerryPy_save/blob/cac5fac60f5e539aec938a343b8152b3587f9ba4/RainsberryPy%20Meteo%20-%20Rapport%20final.pdf')


########################################################################################################################################################################
# Définition de la partie conclusion générale
########################################################################################################################################################################
 
def conclusion():
    st.header("Conclusion")
    '''
    * Notre projet RainsBerryPy nous a permis de mettre en application les différents apprentissages de la formation de Datascientist commencée en octobre 2021 : preprocessing, manipulation de dataframe, DataViz, Machine Learning, interprétabilité, clustering, séries temporelles et même Deep Learning. 

    * Un projet complet qui nous a permis de mettre en avant notre esprit d’initiative en recherchant :
        *   Des éléments nécessaires à notre modélisation : climat de Köppen, circularisation de la variable mois, …
        *   De nouvelles bibliothèques/algorithmes : KNN imputer, Light gbm, Shapash, tslearn, Prophet, FastAI…

    * Aussi, la collaboration au sein de notre groupe s’est très bien déroulée et a démontré que le travail en distanciel (devenue une norme depuis la crise sanitaire) n’entache en rien sa performance.
    * Nous tenions aussi à remercier notre mentor Laurène qui a su questionner notre travail et en assurer sa cohérence à chaque itération.

    ### Pour aller plus loin
    * les possibles évolutions que nous pourrions apporter à notre travail seraient les suivantes :
        * Ajout de variables de géolocalisation de la pluie : pour chaque ligne indiquer la distance de la ville la plus proche où il pleut,
        * Injection de notre résultat de clustering dans notre modèle même si les résultats pourraient être sensiblement similaires à la classification de Koppen,
        * Ajout d’images satellites au jeu de données avec utilisation d’algorithmes de deep learning CNN voir RNN,
        * Utilisation d’algorithme de deep learning RNN sur les séries temporelles.
    '''

########################################################################################################################################################################
# Définition de la partie clustering
########################################################################################################################################################################
    
def clustering():
 
    Menu_mod = st.sidebar.radio(
     "Menu Clustering",
     ('Introduction et stratégie','1ère étape: Type de climat','2ème étape: Régime pluviométrique','3ème étape: Variation de température', 'Conclusion'))  
    
    def Intro():
        st.subheader("Introduction")
        st.image('images/clustering-in-machine-learning.jpg')

        ''' 
        #### La classification de Köppen est une classification des climats fondée sur les précipitations et les températures. Un climat, selon cette classification, est repéré par un code de deux ou trois lettres :
        * 1ère lettre : type de climat 
        * 2ème lettre : régime pluviométrique 
        * 3ème lettre : variations de températures.
        #### La combinaison de ces sous-classifications donne la classification de climat de Köppen suivante :
        '''        
        
        st.image('images/Climat de Koppen.jpg',caption='Classification de Koppen')
        ''' 
        ##### Stratégie Adoptée :
        * 1ère lettre : type de climat => Algorithme KMeans
        * 2ème lettre : régime pluviométrique => TimeSeriesKmeans Clustering
        * 3ème lettre : variations de températures => TimeSeriesKmeans Clustering
        '''                
        
        
    def KMeans():
        st.subheader("Clustering: Type de climat => KMeans")
        '''
        ### Preprocessing:
        #### Création d'un dataframe avec :
        * une ligne par ville
        * pour chaque variable considérée, création d'un jeu de douze colonnes avec le calcul de la moyenne mensuelle: 
            * 'MinTemp','MaxTemp','Temp9am','Temp3pm',
            * 'Rainfall',
            * 'Evaporation',
            * 'Sunshine',
            * 'WindGustSpeed','WindSpeed9am','WindSpeed3pm',
            * 'Humidity9am','Humidity3pm',
            * 'Pressure9am','Pressure3pm',
            * 'Cloud9am','Cloud3pm',
            * 'RainToday_Num'
        ### Utilisation de l'algorithme KMeans:
        #### Méthode du coude pour définir le nombre de clusters
        '''
        st.image('images/1L_Coude.jpg')
        '''
        #### Nous considérons 10 clusters.
        
        ### Comparaison Classification de Koppen vs Clustering 
        '''
        st.image('images/1L_ResultatsTab.jpg')
        '''
        ### Comparaison localisée
        '''
        st.image('images/1L_ResultatsMap.jpg')
        '''
        #### => Climats extrêmes bien identifiés mais résultats moins convaincants pour les autres. 
        '''
    def TSClustering2L():
        st.subheader("Clustering: Régime pluviométrique => TimeSeriesKmeans")
        '''
        ### Preprocessing
        ##### Sélection d'une plage de 3 ans et demi de données à partir de janvier 2014 - Plus grand plages avec des relevés consécutifs (données d'origine avec traitement KNN imputer).

        #### Résultats du Clustering de Séries Temporelles:
        '''
        st.image('images/2L_ResultatsPlot.jpg')
        '''
        ### Comparaison Classification de Koppen vs Clustering
        '''
        st.image('images/21L_ResultatsTab.jpg')
        '''
        ### Comparaison Localisée
        '''        
        st.image('images/2L_ResultatsMap.jpg')
        '''
        ##### => Le régime de mousson est bien isolé et le régime f associé au climat humide se retrouve seul dans de nombreux clusters (hormis 1).
        '''
        
    def TSClustering3L():
        st.subheader("Clustering: Variation de température")
        '''
        ### Preprocessing
        ##### Similaire à la classification précédente

        #### Résultats du Clustering de Séries Temporelles:
        '''
        st.image('images/3L_ResultatsPlot.jpg')
        '''
        ### Comparaison Classification de Koppen vs Clustering
        '''
        st.image('images/3L_ResultatsTab.jpg')
        '''
        ### Comparaison Localisée
        '''
        st.image('images/3L_ResultatsMap.jpg')
        '''
        ##### => L’ensemble des classifications des variations de température est dans l’ensemble bien exécuté.
        '''
    def Conclusion(): 
        st.subheader("Conclusion")
        '''
        ### Combinaison des différents clusters:
        '''
        st.image('images/Clust_ResultatsTab.jpg')
        '''
        #### 32 clusters différents identifiés
        '''
        st.image('images/FinalClust_ResultatsTab.jpg')
        '''
        #### Après regroupement des clusters identifiés sous la même classification de Koppen:
        '''
        st.image('images/Final_ResultatsMap.jpg')
        
    if Menu_mod == 'Introduction et stratégie':
        Intro()
        
    if Menu_mod == '1ère étape: Type de climat':
        KMeans()
        
    if Menu_mod == '2ème étape: Régime pluviométrique':
        TSClustering2L()
        
    if Menu_mod == '3ème étape: Variation de température':
        TSClustering3L()
        
    if Menu_mod == 'Conclusion':
        Conclusion()
        
if __name__ == "__main__":
    main()
