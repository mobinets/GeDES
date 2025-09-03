// #include "traffic_helper.h"
// #include "network_components.h"
// #include <iostream>
// #include "p2p_channel_controller.h"
// #include "switch.h"

// int main1()
// {
//     VDES::create_thread_pool(10);

//     VDES::P2PChannelController p2p_controller;
//     VDES::SwitchController sw_controller;

//     int capacity = 50;
//     VDES::Switch *sw = VDES::CreateSwitch(8, 100, capacity);
//     VDES::Node *n1 = VDES::CreateNode();
//     VDES::Node *n2 = VDES::CreateNode();
//     VDES::P2PChannel *ch1 = VDES::ConnectByP2PChannel(n1, sw, 100, 1000);
//     VDES::P2PChannel *ch2 = VDES::ConnectByP2PChannel(n2, sw, 100, 1000);

//     sw_controller.AddSwitch(sw);
//     p2p_controller.AddChannel(ch1);
//     p2p_controller.AddChannel(ch2);

//     sw_controller.PropertiesAggregation();
//     p2p_controller.PropertiesAggregation();

//     int *p2p_start = new int;
//     int *p2p_end = new int;
//     int p2p_batch_num = 1;
//     *p2p_start = 0;
//     *p2p_end = 2;

//     p2p_controller.FragmentChannels(p2p_start, p2p_end, p2p_batch_num);

//     int *sw_start = new int;
//     int *sw_end = new int;
//     int sw_batch_num = 1;

//     *sw_start = 0;
//     *sw_end = 1;

//     sw_controller.SetBatchInfo(sw_start, sw_end, sw_batch_num);

//     int64_t timeslot_start = 0;
//     int64_t timeslot_end = 0;

//     int64_t *global_timeslot_start;
//     int64_t *global_timeslot_end;
//     cudaMalloc((void **)&global_timeslot_start, sizeof(int64_t));
//     cudaMalloc((void **)&global_timeslot_end, sizeof(int64_t));

//     cudaMemcpy(global_timeslot_start, &timeslot_start, sizeof(int64_t), cudaMemcpyHostToDevice);
//     cudaMemcpy(global_timeslot_end, &timeslot_end, sizeof(int64_t), cudaMemcpyHostToDevice);

//     p2p_controller.SetTimeSlot(global_timeslot_start, global_timeslot_end);
//     sw_controller.SetTimeslot(global_timeslot_start, global_timeslot_end);

//     p2p_controller.BuildCudaGraph();

//     sw_controller.CreateStreams();
//     sw_controller.InitializeBatchIdToStreamId();
//     sw_controller.GenerateKernelParams();
//     sw_controller.BuildGraphs();
//     sw_controller.CreateInstances();

//     return 0;
// }