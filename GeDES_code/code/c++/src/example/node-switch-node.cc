// #include "traffic_helper.h"
// #include "network_components.h"
// #include <iostream>
// #include "p2p_channel_controller.h"
// #include "switch.h"

// /**
//  * @brief: In this case, NIC0 will send to NIC7, NIC1 will send to NIC6, NIC2 will send to NIC5, NIC3 will send to NIC4.
//  */

// enum QUEUE_TYPE
// {
//     FRAME_TYPE = 0,
//     MAC_CACHE_TYPE,
//     IPV4_TYPE,
//     IPV6_TYPE,
//     TCP_TYPE,
//     UDP_TYPE,
// };

// // Timestamp information.
// int64_t timestamp_gap = 1000;

// // global mac addr(For Test Only)
// const int mac_size = 2;
// uint8_t mac_addr[mac_size][6] = {{0, 0, 0, 0, 0, 1}, {0xff, 0xff, 0xff, 0xff, 0xff, 0xff}};

// // global port(For Test Only)
// const int port_size = 5;
// uint16_t ports[] = {0, 1, 2, 3, 4};
// uint32_t mask = 0xffffff00u;

// // A template used for generating random packets.
// static VDES::Frame global_frame = {{1, 1, 1, 1, 1, 1},
//                                    {1, 1, 1, 1, 1, 1},
//                                    {1, 1, 1, 1},
//                                    {1, 1, 1, 1, 1, 1, 1, 1},
//                                    {0, 0, 0, 0, 0, 0, 0, 0},
//                                    {0, 0, 0, 0},
//                                    {3, 3}};

// static VDES::MacCacheEntry global_mac_cache_entry = {{1, 1, 1, 1, 1, 1}, 0};

// /**
//  * @brief: Initialize a node with NICs.
//  * @param node Pointer to the node.
//  * @param capacity Capacity of each queue.
//  */
// void InitializeNode(VDES::Node *node, int size = 90, int capacity = 200);

// /**
//  * @brief: Initialzie a switch with its NICS ,ingress next port and mac cache table.
//  * @param sw Pointer to the switch.
//  * @param capacity Capacity of each queue.
//  */
// void InitializeSwitch(VDES::Switch *sw, int capacity = 200);

// /**
//  * @brief: Generate a frame packet.
//  * @param type Type of the next level packet.
//  */
// VDES::Frame *generate_frame_packet(NetworkProtocolType type = NetworkProtocolType::COUNT_NetworkProtocolType);

// /**
//  * @brief: Generate a mac cache entry.
//  */
// VDES::MacCacheEntry *generate_mac_cache_entry();

// /**
//  * @Generate a queue for cpu.
//  * @param capacity Capacity of the queue.
//  * @param size Size of the queue.
//  */
// template <typename T>
// GPUQueue<T> *malloc_for_cpu(int capacity)
// {
//     GPUQueue<T> *queues = (GPUQueue<T> *)malloc(sizeof(GPUQueue<T>));
//     queues->queue_capacity = capacity;
//     queues->size = 0;
//     queues->queue = (T *)malloc(sizeof(T) * (queues->queue_capacity));
//     queues->head = 0;
//     return queues;
// }

// /**
//  * @brief: Generate multiple cpu queues.
//  * @param queue_num queue num of the queue.
//  * @param capacity Capacity of each queue.
//  */
// template <typename T>
// GPUQueue<T> **malloc_queue_for_cpu(int queue_num, int capacity)
// {
//     GPUQueue<T> **queues = (GPUQueue<T> **)malloc(sizeof(GPUQueue<T> *) * queue_num);
//     for (int i = 0; i < queue_num; i++)
//     {
//         queues[i] = malloc_for_cpu<T>(capacity);
//     }
//     return queues;
// }

// /**
//  * @brief: TYPE OF THE PACKETS :
//  * 0: FRAME
//  * 1: Mac Cache
//  * 2: IPV4
//  * 3: IPV6
//  * 4: TCP
//  * 5: UDP
//  * @param queue_h Pointer to the queue on host.
//  * @param queue_num Capacity of the queue.
//  * @param size Size of the queue.
//  * @param type Type of the packets.
//  */
// void initialize_for_cpu_queue(GPUQueue<void *> *queue_h, int size, QUEUE_TYPE type)
// {
//     queue_h->head = 0;
//     queue_h->size = 0;
//     queue_h->index = 0;
//     for (int i = 0; i < size; i++)
//     {
//         switch (type)
//         {
//             /**
//              * @warning: Must remember that append element will automatically increase the size of the queue.
//              */
//         case FRAME_TYPE:
//             queue_h->append_element(generate_frame_packet());
//             break;
//         case MAC_CACHE_TYPE:
//             queue_h->append_element(generate_mac_cache_entry());
//             break;
//         case IPV4_TYPE:
//             break;
//         case IPV6_TYPE:
//             break;
//         case TCP_TYPE:
//             break;
//         case UDP_TYPE:
//             break;
//         default:
//             break;
//         }
//     }
// }

// /**
//  * @brief Initialize timestamp queue for p2p channel, for this test case,
//  * the generated timestamp will be seperated into 3 parts, such as 0, 1000, 3000,
//  * which correspond to each time slot.
//  * @param queue_h Pointer to the queue on host.
//  * @param capacity Capacity of the queue.
//  * @param size Size of the queue.
//  * @param start_time Start time of the queue.
//  */
// void initialize_for_timestamp(GPUQueue<int64_t> *queue_h, int size, int64_t start_time);

// /**
//  * @brief: Initialize ingress next port queue for switch.
//  * @param ingress_next_port Pointer to the ingress next port queue.
//  * @param capacity Capacity of the queue.
//  */
// void initialize_for_next_port(GPUQueue<uint8_t> *ingress_next_port, int capacity, int port_id);

// /**
//  * @brief: Initialize mac cache queue for switch.
//  * @param mac_cache_cpu Pointer to the mac cache queue on host.
//  * @param capacity Capacity of the queue.
//  */
// void initialize_for_mac_cache(GPUQueue<VDES::MacCacheEntry *> *mac_cache_cpu, int capacity);

// /**
//  * @brief: Copy a single queue from CPU to GPU.
//  * @param queue_h Pointer to the queue on host.
//  * @param queue_capacity Capacity of the queue.
//  * @param size Size of the queue.
//  * @param head Head of the queue.
//  */
// template <typename T>
// void copy_array_to_gpu(GPUQueue<T> *queue_dev, GPUQueue<T> *queue_host)
// {
//     GPUQueue<T> queue_mid;
//     queue_mid.queue_capacity = queue_host->queue_capacity;
//     queue_mid.size = queue_host->size;
//     queue_mid.head = queue_host->head;
//     queue_mid.index = queue_host->index;

//     // cudaMalloc((void **)&(queue_host.queue), sizeof(T) * queue_host->queue_capacity);
//     // Copy the data to the mid queue.
//     cudaMalloc(&(queue_mid.queue), sizeof(T) * queue_host->queue_capacity);
//     cudaMemcpy(queue_mid.queue, queue_host->queue, sizeof(T) * queue_host->queue_capacity, cudaMemcpyHostToDevice);

//     cudaMemcpy(queue_dev, &queue_mid, sizeof(GPUQueue<T>), cudaMemcpyHostToDevice);
// }

// /**
//  * @brief: Copy a queue from CPU to GPU.
//  * @param queue_h Pointer to the queue on host.
//  * @param queue_num Number of queues in the queue.
//  */
// template <typename T>
// GPUQueue<T> **copy_queue_to_gpu(GPUQueue<T> **queue_h, int queue_num)
// {
//     GPUQueue<T> *tmp_queue;

//     GPUQueue<T> **tmp_queues = (GPUQueue<T> **)malloc(sizeof(GPUQueue<T> *) * queue_num);
//     for (int i = 0; i < queue_num; i++)
//     {
//         tmp_queue = create_gpu_queue<T>(queue_h[i]->queue_capacity);
//         // copy data to gpu
//         copy_array_to_gpu<T>(tmp_queue, queue_h[i]);
//         tmp_queues[i] = tmp_queue;
//     }
//     return tmp_queues;
// }

// // 你在调试吗？
// int main()
// {
//     // Gps
//     // Memory pool
//     VDES::InitializeMemoryPools();
//     // Thread pool
//     VDES::create_thread_pool(VDES::THREAD_POOL_SIZE);

//     int capacity = 300;
//     int frame_size = 9;
//     int64_t transmition_rate = 100;
//     int64_t link_delay = 1000;

//     // Set the start time slot and end time slot.
//     int64_t timeslot_start = 0;
//     int64_t timeslot_end = timestamp_gap;

//     int64_t *time_start_gpu;
//     int64_t *time_end_gpu;
//     cudaMalloc(&time_start_gpu, sizeof(int64_t));
//     cudaMalloc(&time_end_gpu, sizeof(int64_t));
//     cudaMemcpy(time_start_gpu, &timeslot_start, sizeof(int64_t), cudaMemcpyHostToDevice);
//     cudaMemcpy(time_end_gpu, &timeslot_end, sizeof(int64_t), cudaMemcpyHostToDevice);

//     // Controllers for running.
//     VDES::P2PChannelController p2p_controller;
//     VDES::SwitchController sw_controller;

//     // Create a switch with 8 ports and 50 capacity.
//     VDES::Switch *sw = VDES::CreateSwitch(8, transmition_rate, capacity);
//     // Initialize Switch.
//     InitializeSwitch(sw, capacity);

//     // Create 2 Nodes, 0-3 for node1, 4-7 for node2.
//     std::vector<VDES::NIC *> nics1;
//     std::vector<VDES::NIC *> nics2;
//     for (int i = 0; i < 4; i++)
//     {
//         nics1.push_back(sw->nics[i]);
//     }
//     for (int i = 4; i < 8; i++)
//     {
//         nics2.push_back(sw->nics[i]);
//     }
//     VDES::Node node1{nics1};
//     VDES::Node node2{nics2};
//     // Initialize Node1 and Node2.
//     InitializeNode(&node1, frame_size, capacity);
//     InitializeNode(&node2, frame_size, capacity);

//     // Create 2 P2PChannel
//     VDES::P2PChannel *ch1 = VDES::ConnectByP2PChannel(&node1, sw, transmition_rate, link_delay);
//     VDES::P2PChannel *ch2 = VDES::ConnectByP2PChannel(&node2, sw, transmition_rate, link_delay);

//     // Add switch to switch controller.
//     sw_controller.AddSwitch(sw);
//     // Add P2P channel to p2p controller and run p2p controller.
//     p2p_controller.AddChannel(ch1);
//     p2p_controller.AddChannel(ch2);

//     sw_controller.PropertiesAggregation();
//     p2p_controller.PropertiesAggregation();

//     // Set the time slot for the switch and p2p channel.
//     sw_controller.SetTimeslot(time_start_gpu, time_end_gpu);
//     p2p_controller.SetTimeSlot(time_start_gpu, time_end_gpu);

//     int *batch_start = new int[1];
//     int *batch_end = new int[1];
//     int batch_num = 1;

//     batch_start[0] = 0;
//     batch_end[0] = 1;

//     sw_controller.SetBatchInfo(batch_start, batch_end, batch_num);

//     int *p2p_batch_start = new int[1];
//     int *p2p_batch_end = new int[1];

//     p2p_batch_start[0] = 0;
//     p2p_batch_end[0] = 2;

//     /** Stage1: Set up the p2p channel controller. */
//     p2p_controller.FragmentChannels(p2p_batch_start, p2p_batch_end, batch_num);
//     p2p_controller.BuildCudaGraph();

//     /** Stage2: Set up the switch controller. */
//     sw_controller.CreateStreams();
//     sw_controller.InitializeBatchIdToStreamId();
//     sw_controller.GenerateKernelParams();
//     sw_controller.BuildGraphs();
//     sw_controller.CreateInstances();

//     /** Stage3: N11 --P2P-- -> N12(On Switch) -> N21(On Switch) --P2P-- -> N22 */
//     auto start_timestamp = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < 3; i++)
//     {
//         p2p_controller.Run();
//         sw_controller.RunAll();
//         // p2p_controller.Run();

//         // cudaDeviceSynchronize();
//         timeslot_start = timeslot_end;
//         timeslot_end += timestamp_gap;
//         sw_controller.UpdateTimeSlot(timeslot_start, timeslot_end);
//         p2p_controller.UpdateTimeSlot(timeslot_start, timeslot_end);
//     }
//     auto end_timestamp = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> last_time = end_timestamp - start_timestamp;

//     std::cout << "Last Time: " << last_time.count() << "s" << std::endl;

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         printf("Cuda error: %s\n", cudaGetErrorString(err));
//         return 1;
//     }

//     // Release all resources.
//     delete sw;

//     // stop thread pool
//     VDES::global_pool->stop();

//     delete VDES::global_pool;

//     std::cout << "Node Switch Node Example Finished." << std::endl;

//     return 0;
// }

// void InitializeNode(VDES::Node *node, int size, int capacity)
// {
//     // Initialize NICs in Node.
//     for (auto *nic : node->nics)
//     {
//         // GPUQueue<VDES::Frame *> *cpu_ingress = malloc_for_cpu<VDES::Frame *>(capacity);
//         // GPUQueue<int64_t> *cpu_ingress_t = malloc_for_cpu<int64_t>(capacity);
//         GPUQueue<VDES::Frame *> *cpu_egress = malloc_for_cpu<VDES::Frame *>(capacity);
//         GPUQueue<int64_t> *cpu_egress_t = malloc_for_cpu<int64_t>(capacity);

//         // initialize_for_cpu_queue<VDES::Frame *>(cpu_ingress, capacity, FRAME_TYPE);
//         // initialize_for_timestamp(cpu_ingress_t, capacity, 0);
//         // In this case, we generate 90 packages, 30 for each time slot.
//         initialize_for_cpu_queue((GPUQueue<void *> *)cpu_egress, size, FRAME_TYPE);
//         initialize_for_timestamp(cpu_egress_t, size, 0);

//         // copy_array_to_gpu<VDES::Frame *>(nic->ingress, cpu_ingress);
//         // copy_array_to_gpu<int64_t>(nic->ingress_t, cpu_ingress_t);
//         copy_array_to_gpu<VDES::Frame *>(nic->egress, cpu_egress);
//         copy_array_to_gpu<int64_t>(nic->egress_t, cpu_egress_t);
//     }
// }

// void InitializeSwitch(VDES::Switch *sw, int capacity)
// {
//     for (int i = 0; i < sw->port_num; i++)
//     {
//         // Initialize NICs in Switch.

//         // Initialize ingress next port.
//         GPUQueue<uint8_t> *ingress_next_port = malloc_for_cpu<uint8_t>(capacity);

//         // initialize_for_next_port(ingress_next_port, capacity, 7 - i);

//         copy_array_to_gpu<uint8_t>(sw->ingress_next_port[i], ingress_next_port);
//     }
//     // Initialize mac cache table.
//     GPUQueue<VDES::MacCacheEntry *> *mac_cache_cpu = malloc_for_cpu<VDES::MacCacheEntry *>(capacity);

//     // initialize_for_cpu_queue((GPUQueue<void *> *)mac_cache_cpu, capacity, MAC_CACHE_TYPE);
//     initialize_for_mac_cache(mac_cache_cpu, capacity);

//     copy_array_to_gpu<VDES::MacCacheEntry *>(sw->mac_table, mac_cache_cpu);
// }

// VDES::Frame *generate_frame_packet(NetworkProtocolType type)
// {
//     if (type == NetworkProtocolType::COUNT_NetworkProtocolType)
//     {
//         type = static_cast<NetworkProtocolType>(rand() % NetworkProtocolType::COUNT_NetworkProtocolType);
//     }

//     memcpy(global_frame.type, &type, sizeof(int16_t));

//     int frame_len = 1500;
//     memcpy(global_frame.frame_len, &frame_len, sizeof(int32_t));

//     /**
//      * Copy a random mac address to the dst_mac field of the frame.
//      */
//     memcpy(global_frame.dst_mac, &mac_addr[rand() % mac_size], sizeof(uint8_t) * 6);
//     VDES::Frame *gpu_frame = VDES::frame_pool->allocate();
//     cudaMemcpy(gpu_frame, &global_frame, sizeof(VDES::Frame), cudaMemcpyHostToDevice);
//     return gpu_frame;
// }

// VDES::MacCacheEntry *generate_mac_cache_entry()
// {
//     int index = rand() % mac_size;

//     // Generate a random mac cache entry.
//     memcpy(global_mac_cache_entry.mac, &mac_addr[index], sizeof(uint8_t) * 6);
//     /**
//      * TODO: Set the specific port for each NIC.
//      */
//     global_mac_cache_entry.port = ports[rand() % port_size];

//     VDES::MacCacheEntry *mac_cache_gpu;
//     cudaMalloc(&mac_cache_gpu, sizeof(VDES::MacCacheEntry));
//     cudaMemcpy(mac_cache_gpu, &global_mac_cache_entry, sizeof(VDES::MacCacheEntry), cudaMemcpyHostToDevice);
//     return mac_cache_gpu;
// }

// void initialize_for_timestamp(GPUQueue<int64_t> *queue_h, int size, int64_t start_time)
// {
//     int64_t time_stamp = start_time;
//     int group = 3, start_size = 0;

//     queue_h->size = size;
//     queue_h->head = 0;
//     queue_h->index = 0;

//     int i = 0;
//     for (int j = 0; j < group; j++)
//     {
//         start_size += size / 3;
//         for (; i < start_size; i++)
//         {
//             queue_h->set_element(i, time_stamp);
//         }
//         time_stamp += 1000;
//     }
//     // Means the end of the transmission.
//     queue_h->set_element(i, time_stamp);
// }

// void initialize_for_next_port(GPUQueue<uint8_t> *ingress_next_port, int capacity, int port_id)
// {
//     for (int i = 0; i < capacity; i++)
//     {
//         /**
//          * TODO: Set the specific port for each NIC.
//          */
//         ingress_next_port->set_element(i, port_id);
//     }
// }

// void initialize_for_mac_cache(GPUQueue<VDES::MacCacheEntry *> *mac_cache_cpu, int capacity)
// {
//     for (int i = 0; i < 8; i++)
//     {
//         if (i < 4)
//         {
//             memcpy(global_mac_cache_entry.mac, &mac_addr[0], sizeof(uint8_t) * 6);
//         }
//         else
//         {
//             memcpy(global_mac_cache_entry.mac, &mac_addr[1], sizeof(uint8_t) * 6);
//         }
//         global_mac_cache_entry.port = 7 - i;

//         VDES::MacCacheEntry *mac_cache_gpu;
//         cudaMalloc(&mac_cache_gpu, sizeof(VDES::MacCacheEntry));
//         cudaMemcpy(mac_cache_gpu, &global_mac_cache_entry, sizeof(VDES::MacCacheEntry), cudaMemcpyHostToDevice);
//     }
// }
