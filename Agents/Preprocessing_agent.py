# Include libraries

class Preprocessing_agent(Agent):

    def __init__(self, aid):
        super(Preprocessing_agent, self).__init__(aid=aid, debug=False)
        self.X = []
        self.Y = []
        self.add = []
        self.encoded_Y = []

    def on_start(self):
        super().on_start()
        self.call_later(1.0, self.preprocessing_agent)

    def preprocessing_agent(self):
        # Loading the dataset pcap format
        data_file = '~/Documents/Traffic_classification/Dataset2H.csv'

        if (data_file):
            df = pd.read_csv(data_file, header=None)

            data_array = df.values
            self.X, self.Y = [], []
            for data in data_array:
                # payload bytes
                self.X.append(data[1:])
                # protocol class
                self.Y.append(data[0])
            self.X = np.array(self.X)
            self.Y = np.array(self.Y)

            # Sampling method 3
            """
            undersampling = NearMiss(version=3, n_neighbors=2)

            # transform the dataset
            self.X, self.Y = undersampling.fit_resample(self.X, self.Y)
            # summarize the new class distribution
            counter = Counter(self.Y)
            print(counter)
            """
            # Sampling method CNN (improved)
            # define the undersampling method
            # define the undersampling method
            undersample = TomekLinks()
            # transform the dataset
            self.X, self.Y = undersample.fit_resample(self.X, self.Y)
            # summarize the new class distribution
            counter = Counter(self.Y)
            print(counter)

        else:
            raise IOError('Non-empty filename expected.')

        ndim = 2
        encoder = LabelEncoder()
        encoder.fit(self.Y)
        self.encoded_Y = encoder.transform(self.Y)

        # getting rid of the problem of unbalanced data set ----- resampling

        print(sorted(Counter(self.Y).items()))
        print("====================================")
        print(sorted(Counter(self.encoded_Y).items()))
        """
        dict_nearMiss = {"bittorrent": 422098, "chat": 2455, "email": 21581, "facebookaudio": 15094,
                         "facebookchart": 16368,
                         "ftp": 191254, "hangoutsaudio": 379726}

        dict_smote = {"bittorrent": 422098, "chat": 2455, "email": 21581, "facebookaudio": 15094,
                      "facebookchart": 16368,
                      "ftp": 191254, "hangoutsaudio": 379726}
        """
        """
        dict_nearMiss = {'ARP':326, 'BROWSER':45, 'CLDAP':36, 'DCERPC':9524, 'DNS':536, 'DRSUAPI':102, 'EPM':82, 'HTTP': 33, 'HTTP/XML': 4, 'IGMPv3': 116, \
            'KRB5': 120, 'LDAP': 847, 'LLMNR': 14, 'LSARPC': 56, 'NBNS': 102, 'NBSS': 15, 'OCSP': 1, 'PIMv2': 240,
                         'RPC_NETLOGON': 2, 'SMB': 20, 'SMB2': 3921, 'SMPP': 960, 'TCP': 214283, 'TLSv1.1': 395, 'TLSv1.2': 93026, 'TLSv1.3': 2155, 'TPKT': 1068, 'UDP': 208}

        dict_smote = {'ARP':326, 'BROWSER':45, 'CLDAP':36, 'DCERPC':9524, 'DNS':536, 'DRSUAPI':102, 'EPM':82, 'HTTP': 33, 'HTTP/XML': 4, 'IGMPv3': 116, \
            'KRB5': 120, 'LDAP': 847, 'LLMNR': 14, 'LSARPC': 56, 'NBNS': 102, 'NBSS': 15, 'OCSP': 1, 'PIMv2': 240,
                         'RPC_NETLOGON': 2, 'SMB': 20, 'SMB2': 3921, 'SMPP': 960, 'TCP': 214283, 'TLSv1.1': 395, 'TLSv1.2': 93026, 'TLSv1.3': 2155, 'TPKT': 1068, 'UDP': 208}

        pipe = make_pipeline(
            SMOTE(sampling_strategy=dict_smote),
            NearMiss(sampling_strategy=dict_nearMiss)
        )

        X_resampled, y_resampled = pipe.fit_resample(self.X, self.Y)

        self.X = X_resampled
        self.Y = y_resampled
        """
        self.encoded_Y = pd.DataFrame(self.encoded_Y, columns=['1'])

        # normalize the dataset

        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler.fit_transform(self.X)

        self.Y = self.encoded_Y.copy()
        self.X = X_train.copy()

        return self.X, self.Y, self.encoded_Y,
        display_message(self.aid.localname, "Hello, I\'m done with preprocessing the dataset!")
