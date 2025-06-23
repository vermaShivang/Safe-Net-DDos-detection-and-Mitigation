import switch
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
from datetime import datetime


class CollectTrainingStatsApp(switch.SimpleSwitch13):
    def __init__(self, *args, **kwargs):
        super(CollectTrainingStatsApp, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self.monitor)

        with open("FlowStatsfile.csv", "w") as file0:
            file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond,label\n')

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info(f'Registered datapath: {datapath.id}')
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.info(f'Unregistered datapath: {datapath.id}')
                del self.datapaths[datapath.id]

    def monitor(self):
        while True:
            if not self.datapaths:
                self.logger.warning("No datapaths connected.")
            for dp in self.datapaths.values():
                self.request_stats(dp)
            hub.sleep(10)

    def request_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        timestamp = datetime.now().timestamp()
        with open("FlowStatsfile.csv", "a") as file0:
            for stat in ev.msg.body:
                if stat.priority != 1:
                    continue

                match = stat.match
                ip_src = match.get('ipv4_src', '0.0.0.0')
                ip_dst = match.get('ipv4_dst', '0.0.0.0')
                ip_proto = match.get('ip_proto', 0)
                tp_src = match.get('tcp_src', match.get('udp_src', 0))
                tp_dst = match.get('tcp_dst', match.get('udp_dst', 0))
                icmp_code = match.get('icmpv4_code', -1)
                icmp_type = match.get('icmpv4_type', -1)

                flow_id = f"{ip_src}{tp_src}{ip_dst}{tp_dst}{ip_proto}"

                try:
                    pps = stat.packet_count / stat.duration_sec if stat.duration_sec else 0
                    ppns = stat.packet_count / stat.duration_nsec if stat.duration_nsec else 0
                    bps = stat.byte_count / stat.duration_sec if stat.duration_sec else 0
                    bpns = stat.byte_count / stat.duration_nsec if stat.duration_nsec else 0
                except:
                    pps = ppns = bps = bpns = 0

                file0.write(f"{timestamp},{ev.msg.datapath.id},{flow_id},{ip_src},{tp_src},{ip_dst},{tp_dst},{ip_proto},{icmp_code},{icmp_type},{stat.duration_sec},{stat.duration_nsec},{stat.idle_timeout},{stat.hard_timeout},{stat.flags},{stat.packet_count},{stat.byte_count},{pps},{ppns},{bps},{bpns},0\n")
