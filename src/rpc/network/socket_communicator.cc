/*!
 *  Copyright (c) 2019 by Contributors
 * \file communicator.cc
 * \brief SocketCommunicator for DGL distributed training.
 */
#include <dmlc/logging.h>

#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <memory>
#include <chrono>
#include <thread>

#include "socket_communicator.h"
#include "../../c_api_common.h"

#ifdef _WIN32
#include <windows.h>
#else   // !_WIN32
#include <unistd.h>
#endif  // _WIN32

namespace dgl {
namespace network {

/////////////////////////////////////// SocketSender ///////////////////////////////////////////

   SocketSender::SocketSender(int64_t queue_size, int max_thread_count)
       : Sender(queue_size, max_thread_count) {
        
     if (max_thread_count_ == 0) {
       max_thread_count_ = 0x8;
     }
     sockets_.resize(max_thread_count_);
     for (int thread_id = 0; thread_id < max_thread_count_; ++thread_id) {
       msg_queue_.push_back(std::make_shared<MessageQueue>(queue_size_));
     }
     for (int thread_id = 0; thread_id < max_thread_count_; ++thread_id) {
       // Create a new thread for this socket connection
       threads_.push_back(std::make_shared<std::thread>(
           &SocketSender::SendLoop, this, thread_id));
     }
     
   }

void SocketSender::AddReceiver(const char* addr, int recv_id) {
  CHECK_NOTNULL(addr);
  if (recv_id < 0) {
    LOG(FATAL) << "recv_id cannot be a negative number.";
  }
  std::vector<std::string> substring;
  std::vector<std::string> ip_and_port;
  SplitStringUsing(addr, "//", &substring);
  // Check address format
  if (substring[0] != "socket:" || substring.size() != 2) {
    LOG(FATAL) << "Incorrect address format:" << addr
               << " Please provide right address format, "
               << "e.g, 'socket://127.0.0.1:50051'. ";
  }
  // Get IP and port
  SplitStringUsing(substring[1], ":", &ip_and_port);
  if (ip_and_port.size() != 2) {
    LOG(FATAL) << "Incorrect address format:" << addr
               << " Please provide right address format, "
               << "e.g, 'socket://127.0.0.1:50051'. ";
  }
  IPAddr address;
  address.ip = ip_and_port[0];
  address.port = std::stoi(ip_and_port[1]);
  receiver_addrs_[recv_id] = address;
}

bool SocketSender::Connect(int recv_id) {
  if (receiver_addrs_.find(recv_id) == receiver_addrs_.end()) {
    LOG(FATAL) << "Cannot find recv_id~" << recv_id;
  }
  int receiver_id = recv_id;
  const auto &addr = receiver_addrs_[recv_id];
  int thread_id = receiver_id % max_thread_count_;
  auto&& sockets_map = sockets_[thread_id];
  std::unique_lock<std::mutex> lk(_mtx);
  sockets_map[receiver_id] = std::make_shared<TCPSocket>();
  TCPSocket *client_socket = sockets_map[receiver_id].get();
  lk.unlock();
  //std::cout<<"~~~~~~~~~~~~~~~ SocketSender::Connect~1"<<std::endl;
  bool bo = false;
  int try_count = 0;
  const char *ip = addr.ip.c_str();
  int port = addr.port;
  while (try_count < kMaxTryCount) {
    if (client_socket->Connect(ip, port)) {
      bo = true;
      //std::cout<<"~~~~~~~~~~~~~~~ SocketSender::Connect~2"<<std::endl;
      break;
    } else {
      if (try_count % 200 == 0 && try_count != 0) {
        // every 1000 seconds show this message
        LOG(INFO) << "Try to connect to: " << ip << ":" << port;
      }
      try_count++;
      //std::cout<<"~~~~~~~~~~~~~~~ SocketSender::Connect~3"<<std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(3));
    }
  }
  return bo;
}

bool SocketSender::Connect() {
  // Create N sockets for Receiver
  /*
  int receiver_count = static_cast<int>(receiver_addrs_.size());
  if (max_thread_count_ == 0 || max_thread_count_ > receiver_count) {
    max_thread_count_ = receiver_count;
  }
  sockets_.resize(max_thread_count_);
  */

  for (const auto& r : receiver_addrs_) {
    int receiver_id = r.first;
    int thread_id = receiver_id % max_thread_count_;
    auto&& sockets_map = sockets_[thread_id];
    std::unique_lock<std::mutex> lk(_mtx);
    sockets_map[receiver_id] = std::make_shared<TCPSocket>();
    TCPSocket* client_socket = sockets_map[receiver_id].get();
    lk.unlock();
    
    bool bo = false;
    int try_count = 0;
    const char* ip = r.second.ip.c_str();
    int port = r.second.port;
    while (bo == false && try_count < kMaxTryCount) {
      if (client_socket->Connect(ip, port)) {
        bo = true;
      } else {
        if (try_count % 200 == 0 && try_count != 0) {
          // every 1000 seconds show this message
          LOG(INFO) << "Try to connect to: " << ip << ":" << port;
        }
        try_count++;
        std::this_thread::sleep_for(std::chrono::seconds(3));
      }
    }
    if (bo == false) {
      return bo;
    }
  }
/*
  for (int thread_id = 0; thread_id < max_thread_count_; ++thread_id) {
    msg_queue_.push_back(std::make_shared<MessageQueue>(queue_size_));
    // Create a new thread for this socket connection
    threads_.push_back(std::make_shared<std::thread>(
      &SocketSender::SendLoop,this,thread_id));
  }
*/
  return true;
}

STATUS SocketSender::Send(Message msg, int recv_id) {
  CHECK_NOTNULL(msg.data);
  CHECK_GT(msg.size, 0);
  CHECK_GE(recv_id, 0);
  msg.receiver_id = recv_id;
  // Add data message to message queue
  STATUS code = msg_queue_[recv_id % max_thread_count_]->Add(msg);
  return code;
}

void SocketSender::Finalize(int recv_id){
  int thread_id = recv_id % max_thread_count_;
  std::unique_lock<std::mutex> lk(_mtx);
    auto&& sockets_map = sockets_[thread_id];
    TCPSocket* client_socket = sockets_map[recv_id].get();
    lk.unlock();
    Message msg;
    msg.size=0;
    SendCore(msg, client_socket);
    client_socket->Close();
    sockets_map.erase(recv_id);
}

void SocketSender::Finalize() {
  _stop = true;
  // Send a signal to tell the msg_queue to finish its job
  for (int i = 0; i < max_thread_count_; ++i) {
    // wait until queue is empty
    auto& mq = msg_queue_[i];
    while (mq->Empty() == false) {
#ifdef _WIN32
        // just loop
#else   // !_WIN32
        usleep(1000);
#endif  // _WIN32
    }
    // All queues have only one producer, which is main thread, so
    // the producerID argument here should be zero.
    mq->SignalFinished(0);
  }
  // Block main thread until all socket-threads finish their jobs
  for (auto& thread : threads_) {
    thread->join();
  }
  // Clear all sockets
  for (auto& group_sockets_ : sockets_) {
    for (auto &socket : group_sockets_) {
      if(socket.second){
      socket.second->Close();
      }
    }
  }
}

void SocketSender::SendCore(Message msg, TCPSocket* socket) {
  if(socket ==nullptr){
    LOG(FATAL)<<" Invalid socket...";
  }
  // First send the size
  // If exit == true, we will send zero size to reciever
  int64_t sent_bytes = 0;
  while (static_cast<size_t>(sent_bytes) < sizeof(int64_t)) {
    int64_t max_len = sizeof(int64_t) - sent_bytes;
    int64_t tmp = socket->Send(
      reinterpret_cast<char*>(&msg.size) + sent_bytes,
      max_len);
    CHECK_NE(tmp, -1);
    sent_bytes += tmp;
  }
  // Then send the data
  sent_bytes = 0;
  while (sent_bytes < msg.size) {
    int64_t max_len = msg.size - sent_bytes;
    int64_t tmp = socket->Send(msg.data+sent_bytes, max_len);
    CHECK_NE(tmp, -1);
    sent_bytes += tmp;
  }
  // delete msg
  if (msg.deallocator != nullptr) {
    msg.deallocator(&msg);
  }
}

void SocketSender::SendLoop(const int id) {
    auto&& sockets = sockets_[id];
    auto&& queue = msg_queue_[id];
  for (;;) {
    std::unique_lock<std::mutex> lk(_mtx);
    bool isEmpty = sockets.size() == 0;
    lk.unlock();
    if(isEmpty && !_stop){
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      continue;
    }

    Message msg;
    STATUS code = queue->Remove(&msg);
    /*
    std::cout << "------- SocketSender::SendLoop remove code: " << code
              << ", msg->size: " << msg.size << " recv_id:" << msg.receiver_id
              << std::endl;
              */
    if (code == QUEUE_CLOSE) {
      msg.size = 0;  // send an end-signal to receiver
      for (auto& socket : sockets) {
        SendCore(msg, socket.second.get());
      }
      break;
    }
    //std::cout<<"-------- msg.receiver_id: "<<msg.receiver_id<<std::endl;
    lk.lock();
    TCPSocket* recv_fd = sockets[msg.receiver_id].get();
    lk.unlock();
    SendCore(msg, recv_fd);
    
  }
}

/////////////////////////////////////// SocketReceiver ///////////////////////////////////////////

bool SocketReceiver::Wait(const char* addr, int num_sender) {
  CHECK_NOTNULL(addr);
  CHECK_GT(num_sender, 0);
  std::vector<std::string> substring;
  std::vector<std::string> ip_and_port;
  SplitStringUsing(addr, "//", &substring);
  // Check address format
  if (substring[0] != "socket:" || substring.size() != 2) {
    LOG(FATAL) << "Incorrect address format:" << addr
               << " Please provide right address format, "
               << "e.g, 'socket://127.0.0.1:50051'. ";
  }
  // Get IP and port
  SplitStringUsing(substring[1], ":", &ip_and_port);
  if (ip_and_port.size() != 2) {
    LOG(FATAL) << "Incorrect address format:" << addr
               << " Please provide right address format, "
               << "e.g, 'socket://127.0.0.1:50051'. ";
  }
  std::string ip = ip_and_port[0];
  int port = stoi(ip_and_port[1]);
  // Initialize message queue for each connection
  num_sender_ = num_sender*4;
#ifdef USE_EPOLL
  if (max_thread_count_ == 0 || max_thread_count_ > num_sender_) {
      max_thread_count_ = num_sender_;
  }
#else
  max_thread_count_ = num_sender_;
#endif
  // Initialize socket and socket-thread
  server_socket_ = new TCPSocket();
  // Bind socket
  if (server_socket_->Bind(ip.c_str(), port) == false) {
    LOG(FATAL) << "Cannot bind to " << ip << ":" << port;
  }

  // Listen
  if (server_socket_->Listen(kMaxConnection) == false) {
    LOG(FATAL) << "Cannot listen on " << ip << ":" << port;
  }

  // start to polling and receiving
  sockets_.resize(max_thread_count_);
  //socket_pool_.resize(max_thread_count_);
  for (int thread_id = 0; thread_id < max_thread_count_; ++thread_id) {
    socket_pool_.emplace_back(SocketPool());
  }
    for (int thread_id = 0; thread_id < max_thread_count_; ++thread_id) {
    // create new thread for each socket pool
    threads_.emplace_back(
        std::make_shared<std::thread>(&SocketReceiver::RecvLoop, this, thread_id));
  }
  //mq_iter_ = msg_queue_.begin();

  // boot a thread for accepting new connections
  threads_.emplace_back(std::make_shared<std::thread>([this]() {
    server_socket_->SetNonBlocking(true);
    while (!stop_accept_) {
      std::string accept_ip;
      int accept_port;
      auto sender_id = curr_num_sender_;
      int thread_id = sender_id % max_thread_count_;
      auto socket = std::make_shared<TCPSocket>();
      if (!server_socket_->Accept(socket.get(), &accept_ip, &accept_port)) {
        // If no connection is acceptable, let's sleep for a while to avoid busy
        // waiting.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      ++curr_num_sender_;
      std::unique_lock<std::mutex> lk(mtx_);
      sockets_[thread_id][sender_id] = socket;
      msg_queue_[sender_id] = std::make_shared<MessageQueue>(queue_size_);
      recv_contexts_[sender_id] =
          std::unique_ptr<RecvContext>(new RecvContext());
      //std::cout<<"~~~~~~~~~ previous size: "<< socket_pool_[thread_id].size()<<std::endl;
      socket_pool_[thread_id].AddSocket(socket, sender_id);
      //std::cout<<"-------------- socket_pool.size: "<<socket_pool_[thread_id].size()<<", thread_id:"<<thread_id<<", type:"<<type_<<std::endl;
      lk.unlock();
      //std::cout<<"--------- New client is accpeted. current num_client: "<< curr_num_sender_ <<
      //  ", thread_id:"<<thread_id<<", sender_id:"<<sender_id<<std::endl;
    }
    std::cout<<"------------- Listening thread in receiver stopped......"<<std::endl;
  }));

  return true;
}

STATUS SocketReceiver::Recv(Message* msg, int* send_id) {
  //std::cout<<"+++++++++++  Receiver::Recv ~ 1 "<<std::endl;
  std::unique_lock<std::mutex> lk(mtx_, std::defer_lock);
  STATUS code = -1;
  bool fetched = false;
  while(true){
    lk.lock();
    for(auto&& p : msg_queue_){
      code = p.second->Remove(msg, false);
      if(code == QUEUE_EMPTY){continue;}
      else{
        *send_id = p.first;
        fetched = true;
        break;
      }
    }
    lk.unlock();
    if(fetched){break;}
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  //std::cout<<"+++++++++++  Receiver::Recv ~ 2 "<<std::endl;
return code;
  /*
  // queue_sem_ is a semaphore indicating how many elements in multiple
  // message queues.
  // When calling queue_sem_.Wait(), this Recv will be suspended until
  // queue_sem_ > 0, decrease queue_sem_ by 1, then start to fetch a message.
  //std::cout<<"------ SocketReceiver::Recv~1"<<std::endl;
  queue_sem_.Wait();
  //std::cout<<"------ SocketReceiver::Recv~2"<<std::endl;
  for (;;) {
    for (; mq_iter_ != msg_queue_.end(); ++mq_iter_) {
      STATUS code = mq_iter_->second->Remove(msg, false);
      if (code == QUEUE_EMPTY) {
        continue;  // jump to the next queue
      } else {
        *send_id = mq_iter_->first;
        ++mq_iter_;
        return code;
      }
    }
    //std::cout<<"------ SocketReceiver::Recv~3"<<std::endl;
    //std::this_thread::sleep_for(std::chrono::milliseconds(100));
    mq_iter_ = msg_queue_.begin();
  }
  //std::cout<<"------ SocketReceiver::Recv~4"<<std::endl;
  */
}

STATUS SocketReceiver::RecvFrom(Message* msg, int send_id) {
  // Get message from specified message queue
  //queue_sem_.Wait();
  //std::cout<<"+++++++++++  Receiver::RecvFrom ~ 1 "<<std::endl;
  std::unique_lock<std::mutex> lk(mtx_);
  auto&& mq = msg_queue_[send_id];
  lk.unlock();
  STATUS code = mq->Remove(msg);
  //std::cout<<"+++++++++++  Receiver::RecvFrom ~ 2 "<<std::endl;
  return code;
}

void SocketReceiver::Finalize(int sender_id){
  int thread_id = sender_id % max_thread_count_;
  std::lock_guard<std::mutex> lk(mtx_);
      auto&& socket = sockets_[thread_id][sender_id];
      socket->Close();
      msg_queue_.erase(sender_id);
      recv_contexts_.erase(sender_id);
      socket_pool_[thread_id].RemoveSocket(socket);
      sockets_[thread_id].erase(sender_id);
}

void SocketReceiver::Finalize() {
  std::cout<<"-------------- SocketReceiver::Finalize()~1 -------------"<<std::endl;
  // Send a signal to tell the message queue to finish its job
  for (auto& mq : msg_queue_) {
    // wait until queue is empty
    while (mq.second->Empty() == false) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    mq.second->SignalFinished(mq.first);
  }
std::cout<<"-------------- SocketReceiver::Finalize()~2 -------------"<<std::endl;
  // stop accept thread
  stop_accept_ = true;

  // Block main thread until all socket-threads finish their jobs
  for (auto& thread : threads_) {
    thread->join();
  }
  std::cout<<"-------------- SocketReceiver::Finalize()~3 -------------"<<std::endl;
  // Clear all sockets
  for (auto& group_sockets : sockets_) {
    for (auto& socket : group_sockets) {
      socket.second->Close();
    }
  }
  std::cout<<"-------------- SocketReceiver::Finalize()~4 -------------"<<std::endl;
  server_socket_->Close();
  delete server_socket_;
}

int64_t RecvDataSize(TCPSocket* socket) {
  int64_t received_bytes = 0;
  int64_t data_size = 0;
  while (static_cast<size_t>(received_bytes) < sizeof(int64_t)) {
    int64_t max_len = sizeof(int64_t) - received_bytes;
    int64_t tmp = socket->Receive(
      reinterpret_cast<char*>(&data_size) + received_bytes,
      max_len);
    if (tmp == -1) {
      if (received_bytes > 0) {
        // We want to finish reading full data_size
        continue;
      }
      return -1;
    }
    received_bytes += tmp;
  }
  return data_size;
}

void RecvData(TCPSocket* socket, char* buffer, const int64_t &data_size,
  int64_t *received_bytes) {
  while (*received_bytes < data_size) {
    int64_t max_len = data_size - *received_bytes;
    int64_t tmp = socket->Receive(buffer + *received_bytes, max_len);
    if (tmp == -1) {
      // Socket not ready, no more data to read
      return;
    }
    *received_bytes += tmp;
  }
}

void SocketReceiver::RecvLoop( const int thread_id) {
  //std::this_thread::sleep_for(std::chrono::milliseconds(300));//This line works
  
  //std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  // Main loop to receive messages
  while(!stop_accept_) {
    //if (type_==1) std::cout<<"-------!!!!! RecvLoop~00000~ thread_id:"<<thread_id<<std::endl;
    std::unique_lock<std::mutex> lk(mtx_);
    auto&& socket_pool = socket_pool_[thread_id];
    auto&& queues = msg_queue_;
    //auto&& queue_sem = &queue_sem_;
    lk.unlock();
    if (type_ == 1){
    ;//std::cout<<"-------------- RecvLoop::socket_pool.size: "<<socket_pool.size()<<", thread_id:"<<thread_id<<", type:"<<type_<<std::endl;
    }
    int sender_id;
    // Get active socket using epoll
    std::shared_ptr<TCPSocket> socket = socket_pool.GetActiveSocket(&sender_id);
    if(!socket){
      //no active socket
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      continue;
    }
    //if (type_==1) std::cout<<"-------!!!!! RecvLoop~0~sender_id:"<<sender_id<<", thread_id:"<<thread_id<<std::endl;
    if (queues[sender_id]->EmptyAndNoMoreAdd()) {
      if(type_==1) std::cout<<"----------- sender_id stopped....."<<std::endl;
      // This sender has already stopped
      if (socket_pool.RemoveSocket(socket) == 0) {
        continue;//return;
      }
      continue;
    }
    //if (type_==1) std::cout<<"------- RecvLoop~1~sender_id:"<<sender_id<<", thread_id:"<<thread_id<<std::endl;
    // Nonblocking socket might be interrupted at any point. So we need to
    // store the partially received data
    lk.lock();
    std::unique_ptr<RecvContext> &ctx = recv_contexts_[sender_id];
    lk.unlock();
    int64_t &data_size = ctx->data_size;
    int64_t &received_bytes = ctx->received_bytes;
    char*& buffer = ctx->buffer;

    if (data_size == -1) {
      // This is a new message, so receive the data size first
      data_size = RecvDataSize(socket.get());
      if (data_size > 0) {
        try {
          buffer = new char[data_size];
        } catch(const std::bad_alloc&) {
          LOG(FATAL) << "Cannot allocate enough memory for message, "
                     << "(message size: " << data_size << ")";
        }
        received_bytes = 0;
      } else if (data_size == 0) {
        // Received stop signal
        if (socket_pool.RemoveSocket(socket) == 0) {
          std::cout<<"!!!!!!!!!! data_size~0:"<<data_size<<std::endl;
          continue;//return;
        }
        else{
          std::cout<<"!!!!!!!!!! NOT EXPECTED data_size:"<<data_size<<std::endl;
          continue;
        }
      }
      else{
        std::cout<<"!!!!!!!!!! NOT Not EXPECTED data_size:"<<data_size<<std::endl;
        continue;
      }
    }
//if (type_==1) std::cout<<"------- RecvLoop~2~sender_id:"<<sender_id<<", thread_id:"<<thread_id<<std::endl;
    RecvData(socket.get(), buffer, data_size, &received_bytes);
    if (received_bytes >= data_size) {
      // Full data received, create Message and push to queue
      Message msg;
      msg.data = buffer;
      msg.size = data_size;
      msg.deallocator = DefaultMessageDeleter;
      lk.lock();
      queues[sender_id]->Add(msg);
      lk.unlock();

      // Reset recv context
      data_size = -1;

      // Signal queue semaphore
      //if(type_==1) std::cout<<"--------- RecvLoop::sem_post(), sender_id:"<<sender_id<<std::endl;
      //queue_sem->Post();
    }
  }
  if(type_==1) std::cout<<"!!!!!!!!!!!!!!!!!! RecvLoop is exiting........."<<std::endl;
}

}  // namespace network
}  // namespace dgl
