"""Functions used by server."""

import time

from . import rpc
from .constants import MAX_QUEUE_SIZE, SERVER_EXIT, SERVER_KEEP_ALIVE

def start_server(server_id, ip_config, num_servers, num_clients, server_state, \
    max_queue_size=MAX_QUEUE_SIZE, net_type='socket'):
    """Start DGL server, which will be shared with all the rpc services.

    This is a blocking function -- it returns only when the server shutdown.

    Parameters
    ----------
    server_id : int
        Current server ID (starts from 0).
    ip_config : str
        Path of IP configuration file.
    num_servers : int
        Server count on each machine.
    num_clients : int
        Total number of clients that will be connected to the server.
        Note that, we do not support dynamic connection for now. It means
        that when all the clients connect to server, no client will can be added
        to the cluster.
    server_state : ServerSate object
        Store in main data used by server.
    max_queue_size : int
        Maximal size (bytes) of server queue buffer (~20 GB on default).
        Note that the 20 GB is just an upper-bound because DGL uses zero-copy and
        it will not allocate 20GB memory at once.
    net_type : str
        Networking type. Current options are: 'socket'.
    """
    assert server_id >= 0, 'server_id (%d) cannot be a negative number.' % server_id
    assert num_servers > 0, 'num_servers (%d) must be a positive number.' % num_servers
    assert num_clients >= 0, 'num_client (%d) cannot be a negative number.' % num_client
    assert max_queue_size > 0, 'queue_size (%d) cannot be a negative number.' % queue_size
    assert net_type in ('socket'), 'net_type (%s) can only be \'socket\'' % net_type
    # Register signal handler.
    rpc.register_sig_handler()
    # Register some basic services
    rpc.register_service(rpc.CLIENT_REGISTER,
                         rpc.ClientRegisterRequest,
                         rpc.ClientRegisterResponse)
    rpc.register_service(rpc.SHUT_DOWN_SERVER,
                         rpc.ShutDownRequest,
                         None)
    rpc.register_service(rpc.GET_NUM_CLIENT,
                         rpc.GetNumberClientsRequest,
                         rpc.GetNumberClientsResponse)
    rpc.register_service(rpc.CLIENT_BARRIER,
                         rpc.ClientBarrierRequest,
                         rpc.ClientBarrierResponse)
    rpc.set_rank(server_id)
    server_namebook = rpc.read_ip_config(ip_config, num_servers)
    machine_id = server_namebook[server_id][0]
    rpc.set_machine_id(machine_id)
    ip_addr = server_namebook[server_id][1]
    port = server_namebook[server_id][2]
    rpc.create_sender(max_queue_size, net_type)
    rpc.create_receiver(max_queue_size, net_type, 1)
    # wait all the senders connect to server.
    # Once all the senders connect to server, server will not
    # accept new sender's connection
    print("Wait connections ...")
    rpc.receiver_wait(ip_addr, port, num_clients)
    
    # main service loop
    addr_list = {}
    
    ready_to_init = []
    ready_group_id = 0
    curr_init_client_num = 4
    inited_client_map={}
    while True:
        if len(ready_to_init) > 0:
            #hande-shake logic
            print("------------- group_id:{}, num_clients:{} connected!!!!!".format(ready_group_id,num_clients))
            rpc.set_num_client(num_clients, ready_group_id)
            # Recv all the client's IP and assign ID to clients
            ready_to_init.sort()
            client_namebook = {}
            for client_id, addr in enumerate(ready_to_init):
                client_namebook[client_id] = addr
            assert ready_group_id not in inited_client_map
            inited_client_map[ready_group_id] = {}
            for client_id, addr in client_namebook.items():
                client_ip, client_port = addr.split(':')
                g_client_id = ready_group_id*100+client_id
                rpc.add_receiver_addr(client_ip, client_port, g_client_id)
                inited_client_map[ready_group_id][client_id] = (g_client_id, addr)
                rpc.record_group_client_id(ready_group_id, g_client_id)
                rpc.sender_connect(g_client_id)
                print("----- Sender is connected to recv: g_client_id~{}, client_id~{}, client_ip~{}, client_port~{}".format(g_client_id,client_id, client_ip, client_port))
            
            #####rpc.sender_connect_orig()
            time.sleep(3) # required @21.11.12 to make sure client has called receiver.wait()
            if rpc.get_rank() == 0: # server_0 send all the IDs
                for client_id, _ in client_namebook.items():
                    register_res = rpc.ClientRegisterResponse(client_id)
                    g_client_id, _ = inited_client_map[ready_group_id][client_id]
                    rpc.send_response(g_client_id, register_res)
                    print("--------- ClientRegisterResponse is sent to {}".format(g_client_id))
            ready_to_init=[]
            ready_group_id = 0

        req, client_id = rpc.recv_request()

        if isinstance(req, rpc.ClientRegisterRequest):
            #print("--------- ClientRegisterRequest.ip_addr: {}, group_id: {}".format(req.ip_addr, req.group_id))
            if req.group_id not in addr_list:
                addr_list[req.group_id] = []
            addr_list[req.group_id].append(req.ip_addr)
            #rpc.record_recv_group_client_id(req.group_id, client_id)
            #print("-------------- client_id/sender_id in receiver: {}".format(client_id))
            #print("**************** group_id:{}, num:{}".format(req.group_id, len(addr_list[req.group_id])))
            assert len(addr_list[req.group_id]) <= num_clients
            if len(addr_list[req.group_id]) == num_clients:
                ready_to_init = addr_list[req.group_id]
                ready_group_id = req.group_id
            continue
        
        res = req.process_request(server_state)
        if res is not None:
            if isinstance(res, list):
                for response in res:
                    target_id, res_data = response
                    target_id, _ = inited_client_map[req.group_id][target_id]
                    rpc.send_response(target_id, res_data)
            elif isinstance(res, str):
                if res == SERVER_EXIT:
                    # exit server process
                    return
                elif res == SERVER_KEEP_ALIVE:
                    # keep alive
                    print("-------------- Server keeps alive...........")
                    continue
                else:
                    raise RuntimeError("Unexpected return code: {}".format(res))
            else:
                client_id, _ = inited_client_map[req.group_id][req.client_id]
                rpc.send_response(client_id, res)
                msg = res.msg if hasattr(res, 'msg') else "Unknown"
                print("------------ SERVER response g_client_id: {}, msg:{}, client_id~{}, group~{}".format(client_id, msg,req.client_id, req.group_id))
