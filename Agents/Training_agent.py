# Include libraries here

class Training_agent(Agent):

    def __init__(self, aid):
        super(Training_agent, self).__init__(aid=aid, debug=False)

    def on_start(self):
        super().on_start()
        self.call_later(1.0, self.training_agent)

    def training_agent(self):
        # get the start time
        stt = time.time()
        X = preprocessingagent.X
        Y = preprocessingagent.Y
        encoded_Y = preprocessingagent.encoded_Y

        stt2 = time.time()

        stt3 = stt2 - etp1
        print('Time - Preprocessing - Training agents:', stt3, 'seconds')

        # 4H15: 41/40
        """
        LABELS = ['ARP', 'BROWSER', 'CLDAP', 'DCERPC', 'DNS', 'DRSUAPI', 'EPM', 'Gearman',  'HTTP',  'HTTP/XML',  'ICMP', 'IGMPv3', 'IRC', 'KRB5',
                  'LDAP', 'LLMNR', 'LSARPC', 'NBNS', 'NBSS', 'NTP', 'OCSP', 'PIMv2', 'RDP', 'RPC_NETLOGON', 'RTMP',  'SAMR', 'SMB',
                  'SMB2', 'SMPP', 'SNMP', 'SSL', 'SSLv2', 'TCP', 'TLSv1.1', 'TLSv1.2', 'TLSv1.3', 'TPKT', 'UDP', 'WINREG', 'X11']
        """

        # 2 H: 29
        LABELS = ['ARP', 'BROWSER', 'CLDAP', 'DCERPC', 'DNS', 'DRSUAPI', 'EPM', 'HTTP', 'HTTP/XML', 'IGMPv3', 'KRB5',
                  'LDAP', 'LLMNR', 'LSARPC',
                  'NBNS', 'NBSS', 'NTP', 'PIMv2', 'RDP', 'RPC_NETLOGON', 'SMB', 'SMB2', 'SMPP', 'TCP', 'TLSv1.1',
                  'TLSv1.2', 'TLSv1.3', 'TPKT', 'UDP']

        # 1H:30: 103
        """
        LABELS = ['ADP', 'AODV', 'ARP', 'ASAP', 'AX4000', 'Auto-RP', 'BFD Control', 'BFD Echo', 'BJNP', 'BOOTP', 'BROWSER',
         'CIP I/O', 'CLDAP', 'CN/IP', 'CUPS', 'Chargen', 'DAYTIME', 'DB-LSP-DISC/JSON', 'DCERPC', 'DHCPv6', 'DNPv13',
         'DNS', 'DPNET', 'DRSUAPI', 'DTLS', 'ECHO', 'ENRP', 'EPM', 'FIND', 'H.225.0', 'H.248', 'HCrt', 'HTTP',
         'HTTP/XML', 'IAPP', 'ICMP', 'ICP', 'IGMPv3', 'IPX', 'IPv6', 'ISAKMP', 'KNET', 'KPASSWD', 'KRB5', 'LDAP', 'LLC',
         'LLMNR', 'LSARPC', 'LWAPP', 'MANOLITO', 'MDNS', 'MSproxy', 'MobileIP', 'NBDS', 'NBNS', 'NCP', 'NTP', 'OCSP',
         'OpenVPN', 'PIMv2', 'PKTC', 'QUAKE', 'QUAKE2', 'QUAKE3', 'QUAKEWORLD', 'RDPUDP', 'RIP', 'RIPng', 'RPC',
         'RPC_NETLOGON', 'RSIP', 'RSVP', 'SABP', 'SAP', 'SEBEK', 'SMB', 'SMB2', 'SMPP', 'SNMP', 'SRVLOC', 'Syslog',
         'TCP', 'TETRA', 'TIME', 'TLSv1.1', 'TLSv1.2', 'TLSv1.3', 'TPCP', 'TPKT', 'TPLINK-SMARTHOME/JSON', 'TSP', 'TZSP',
         'UDP', 'UDPENCAP', 'ULP', 'Vuze-DHT', 'WHO', 'WLCCP', 'WSP', 'WTLS+WSP',  'WTLS + WTP + WSP',  'WTP + WSP', 'XYPLEX']
         """

        # 1H: 28
        """
        LABELS = ['ARP', 'BROWSER', 'CLDAP', 'DCERPC', 'DNS', 'DRSUAPI', 'EPM', 'HTTP', 'HTTP/XML', 'IGMPv3', \
            'KRB5', 'LDAP', 'LLMNR', 'LSARPC', 'NBNS', 'NBSS', 'OCSP', 'PIMv2',
                         'RPC_NETLOGON', 'SMB', 'SMB2', 'SMPP', 'TCP', 'TLSv1.1', 'TLSv1.2', 'TLSv1.3' , 'TPKT', 'UDP']
        """

        # 30 M: 48
        """
        LABELS = ['ARP', 'BROWSER', 'CLDAP', 'DCERPC', 'DNS', 'DRSUAPI', 'EPM', 'IGMPv3', 'KRB5', 'LDAP', 'LLMNR', 'LSARPC', 'NBNS', 'PIMv2',
                  'RPC_NETLOGON', 'SMB', 'SMB2', 'SMPP', 'TCP', 'TLSv1.1', 'TLSv1.2', 'TLSv1.3', 'TPKT', 'UDP', ]
        """

        # self.LABELS = LABELS

        counter = iter(range(2))
        activation = 'relu'
        class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(LABELS), y=LABELS)
        class_weights = dict(enumerate(class_weights))
        print(class_weights)

        num_classes = 29  # len(LABELS)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                            stratify=encoded_Y, random_state=10)
        X_train_ann = X_train.copy()
        X_test_ann = X_test.copy()
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        Y_train_ann_cnn = to_categorical(Y_train, num_classes)
        Y_test_ann_cnn = to_categorical(Y_test, num_classes)
        print("===========================")
        print(X_train_ann.shape, X_test_ann.shape, Y_train_ann_cnn.shape, Y_test_ann_cnn.shape,
              X_train_cnn.shape,
              X_test_cnn.shape)
        """
        # ANN MODEL
        model = Sequential()
        model.add(Dense(1024, input_dim=2, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        """
        # CNN 1

        model = Sequential()
        model.add(Conv1D(filters=2, kernel_size=2, activation='relu', input_shape=(2, 1), padding='same'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(2048, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        # CNN2

        # This function will be used during evaluation

        def get_f1(y_true, y_pred):  # taken from old keras source code
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            recall = true_positives / (possible_positives + K.epsilon())
            f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
            return f1_val

        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['acc', keras.metrics.Precision(), keras.metrics.Recall()])

        # rmsprop, Adamax, Adam, Nadam

        history = model.fit(X_train_ann, Y_train_ann_cnn, verbose=1, epochs=1, batch_size=64,
                            validation_data=(X_test_ann, Y_test_ann_cnn), class_weight=class_weights)

        loss, accuracy, precision, recall = model.evaluate(X_test_ann, Y_test_ann_cnn, verbose=1)

        # Save the model to use during the Prediction
        model.save('~/Documents/Traffic_classification/MyNewModel_h5', save_format='h5')
        print("Accuracy:")
        print(accuracy)
        print("Precision:")
        print(precision)
        print("Recall:")
        print(recall)
        # print("F1:")
        # print(f1)

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['Training', 'Testing'], loc='upper left')
        plt.savefig("Accuracy_img.png")

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['Training', 'Testing'], loc='upper left')
        plt.savefig("Loss_img.png")
        # get the end time
        ett = time.time()
        # get the execution time
        elapsed_timet = ett - stt
        print('Execution time for the training agent:', elapsed_timet, 'seconds')

        self.X_test_ann = X_test.copy()
        self.Y_test_ann_cnn = to_categorical(Y_test, num_classes)
        self.LABELS = LABELS

        self.elapsed_timet = elapsed_timet

        return self.X_test_ann, self.Y_test_ann_cnn, self.LABELS, self.elapsed_timet

        display_message(self.aid.localname, "Hello, I\'m done with the training process!")
