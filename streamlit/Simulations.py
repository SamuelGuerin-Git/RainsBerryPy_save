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

    st.subheader("Interprétabilité")
    
    #if st.button("Importance des features"):
    #    picklefile = open("modeles/xgboost.pkl", "rb")
    #    modele = pickle.load(picklefile)  
    #    explainer = shap.TreeExplainer(modele)
    #    shap_values = explainer.shap_values(df[features])
    #    st_shap(shap.summary_plot(shap_values, df[features]),height=300)
