s# Include libraries here

class  Prediction_agent(Agent):

    def __init__(self, aid):
        super(Prediction_agent, self).__init__(aid=aid, debug=False)

    def on_start(self):
        super().on_start()
        self.call_later(1.0, self.prediction_agent)

    def prediction_agent(self):
        # get the start time
        stpr = time.time()

        self.X_test_ann = trainingagent.X_test_ann
        self.Y_test_ann_cnn = trainingagent.Y_test_ann_cnn
        self.LABELS = trainingagent.LABELS

        stpr2 = time.time()

        sttpr3 = stpr2 - stpr
        print('Time Training - Prediction agents:', sttpr3, 'seconds')

        # loading the saved model
        model = tf.keras.models.load_model('~/Documents/Traffic_classification/MyNewModel_h5')
        print("For Test Data: ")
        Y_pred = model.predict(self.X_test_ann)
        print("Confusion Matrix:")
        matrix = confusion_matrix(self.Y_test_ann_cnn.argmax(axis=1), Y_pred.argmax(axis=1))
        for i in range(len(matrix)):
            k = matrix[i, :]
            for j in k:
                print(j, end=" ")
            print("")
        """
        # 1H
        for i in range(28):
            print(self.LABELS[i], ": ", (matrix[i, i] / sum(matrix[i, :])) * 100, "%")
        """

        # 30 M
        """"""
        #for i in range(29):
             #print(self.LABELS[i], ": ", (matrix[i, i] / sum(matrix[i, :])) * 100, "%")

        # 1H30M

        for i in range(29):
            print(self.LABELS[i], ": ", (matrix[i, i] / sum(matrix[i, :])) * 100, "%")

        matrix = pd.DataFrame(matrix, index=self.LABELS, columns=self.LABELS)
        plt.figure(figsize=(10, 28))
        sn.heatmap(matrix, annot=True)
        plt.show()

        print("For Test Data: Full Classification Report ")
        Y_test = np.argmax(self.Y_test_ann_cnn, axis=1)  # Convert one-hot to index
        # y_pred = model.predict_classes(X_test_ann)
        y_pred = np.argmax(model.predict(self.X_test_ann), axis=1)
        print(classification_report(Y_test, y_pred))

        # get the end

        etpr = time.time()
        # get the execution time
        elapsed_timepr = etpr - stpr
        print('Execution time for the prediction agent:', elapsed_timepr, 'seconds')

        display_message(self.aid.localname, "Hello, I\'m done with the prediction process!")