import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.lib import hub
from datetime import datetime

import switchm  # Your existing switch class


class SimpleMonitorCNN(switchm.SimpleSwitch13):
    def __init__(self, *args, **kwargs):
        super(SimpleMonitorCNN, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)

        self.mitigation = 0
        self.scaler = StandardScaler()

        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        print("Training time: ", (end - start))

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)
            self.flow_predict()

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        timestamp = datetime.now().timestamp()

        with open("PredictFlowStatsfile.csv", "w") as file0:
            file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
            body = ev.msg.body
            for stat in sorted([flow for flow in body if (flow.priority == 1)], key=lambda flow: (flow.match['eth_type'], flow.match['ipv4_src'], flow.match['ipv4_dst'], flow.match['ip_proto'])):
                ip_src = stat.match['ipv4_src']
                ip_dst = stat.match['ipv4_dst']
                ip_proto = stat.match['ip_proto']
                icmp_code = -1
                icmp_type = -1
                tp_src = 0
                tp_dst = 0

                if ip_proto == 1:
                    icmp_code = stat.match.get('icmpv4_code', -1)
                    icmp_type = stat.match.get('icmpv4_type', -1)
                elif ip_proto == 6:
                    tp_src = stat.match.get('tcp_src', 0)
                    tp_dst = stat.match.get('tcp_dst', 0)
                elif ip_proto == 17:
                    tp_src = stat.match.get('udp_src', 0)
                    tp_dst = stat.match.get('udp_dst', 0)

                flow_id = f"{ip_src}{tp_src}{ip_dst}{tp_dst}{ip_proto}"

                try:
                    packet_count_per_second = stat.packet_count / stat.duration_sec
                    packet_count_per_nsecond = stat.packet_count / stat.duration_nsec
                except:
                    packet_count_per_second = 0
                    packet_count_per_nsecond = 0

                try:
                    byte_count_per_second = stat.byte_count / stat.duration_sec
                    byte_count_per_nsecond = stat.byte_count / stat.duration_nsec
                except:
                    byte_count_per_second = 0
                    byte_count_per_nsecond = 0

                file0.write(f"{timestamp},{ev.msg.datapath.id},{flow_id},{ip_src},{tp_src},{ip_dst},{tp_dst},{ip_proto},{icmp_code},{icmp_type},{stat.duration_sec},{stat.duration_nsec},{stat.idle_timeout},{stat.hard_timeout},{stat.flags},{stat.packet_count},{stat.byte_count},{packet_count_per_second},{packet_count_per_nsecond},{byte_count_per_second},{byte_count_per_nsecond}\n")

    def flow_training(self):
        
        print(tf.config.list_physical_devices('GPU'))

        self.logger.info("Flow Training ...")
        try:
            data = pd.read_csv('dataset.csv')

            # Preprocess columns (replace dots)
            for col in [2, 3, 5]:  # flow_id, ip_src, ip_dst
                data.iloc[:, col] = data.iloc[:, col].astype(str).str.replace('.', '')

            X = data.iloc[:, :-1].values.astype('float64')
            y = data.iloc[:, -1].values.astype('int')

            # Scale features
            X = self.scaler.fit_transform(X)

            # Reshape for CNN (samples, features, 1)
            X = np.expand_dims(X, axis=2)

            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

            # Build CNN model
            model = models.Sequential([
                layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(64, kernel_size=3, activation='relu'),
                layers.MaxPooling1D(pool_size=2),
                layers.Flatten(),
                layers.Dense(100, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            # Train model
            model.fit(X_train[:500], y_train[:500], epochs=100, batch_size=32, verbose=1)

            # Evaluate
            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            self.logger.info(f"Training completed with accuracy: {acc:.4f}")

            # Predict and show confusion matrix
            y_pred_prob = model.predict(X_test)
            y_pred = (y_pred_prob > 0.5).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            self.logger.info("Confusion Matrix:")
            self.logger.info(cm)

            self.flow_model = model

        except Exception as e:
            self.logger.error(f"Error in training: {e}")

    def flow_predict(self):
        try:
            data = pd.read_csv('PredictFlowStatsfile.csv')

            for col in [2, 3, 5]:
                data.iloc[:, col] = data.iloc[:, col].astype(str).str.replace('.', '')

            X = data.values.astype('float64')
            X = self.scaler.transform(X)
            X = np.expand_dims(X, axis=2)

            y_pred_prob = self.flow_model.predict(X)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()

            legitimate_count = np.sum(y_pred == 0)
            attack_count = np.sum(y_pred == 1)

            self.logger.info("------------------------------------------------------------------------------")
            total_flows = len(y_pred)
            if total_flows == 0:
                self.logger.info("No flows detected.")
                return

            if (legitimate_count / total_flows) > 0.8:
                self.logger.info("Traffic is Legitimate!")
                self.mitigation = 0
            else:
                victim = None
                # Pick victim host from first detected attack flow
                attack_idx = np.where(y_pred == 1)[0]
                if len(attack_idx) > 0:
                    victim_ip_raw = int(data.iloc[attack_idx[0], 5])
                    victim = victim_ip_raw % 20
                self.logger.info("NOTICE!! DoS Attack in Progress!!!")
                if victim is not None:
                    self.logger.info(f"Victim Host: h{victim}")
                print("Mitigation process in progress!")
                self.mitigation = 1
            self.logger.info("------------------------------------------------------------------------------")

            # Clear the file after prediction
            with open("PredictFlowStatsfile.csv", "w") as file0:
                file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')

        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")

