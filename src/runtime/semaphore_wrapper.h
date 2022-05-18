/*!
 *  Copyright (c) 2021 by Contributors
 * \file semaphore_wrapper.h
 * \brief A simple corss platform semaphore wrapper
 */
#ifndef DGL_RUNTIME_SEMAPHORE_WRAPPER_H_
#define DGL_RUNTIME_SEMAPHORE_WRAPPER_H_

#ifdef _WIN32
#include <windows.h>
#else
#include <semaphore.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#endif
#include <dmlc/logging.h>


namespace dgl {
namespace runtime {

/*!
 * \brief A simple crossplatform Semaphore wrapper
 */
class Semaphore {
 public:
  /*!
   * \brief Semaphore constructor
   */
  Semaphore();
  /*!
   * \brief blocking wait, decrease semaphore by 1
   */
  void Wait();
  bool TimedWait(int timeout) {
#ifdef _WIN32
    Wait();
    return true;
#else
    if (timeout == -1) {
      timeout = 3600 * 24 * 1000;
    }
    timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
      LOG(ERROR) << "Failed to call clock_gettime...";
      return false;
    }
    int nsec = std::max(timeout / 1000, 1);
    ts.tv_sec += nsec;
    int ret = 0;
    while ((ret = sem_timedwait(&sem_, &ts) != 0) && errno == EINTR) {
      continue;
    };
    if (ret != 0 && errno == ETIMEDOUT) {
      LOG(ERROR) << "sem_timedwait timeout after " << timeout
                 << " milliseconds.";
      return false;
    }
    return ret == 0;
#endif
  }
  /*!
   * \brief increase semaphore by 1
   */
  void Post();
 private:
#ifdef _WIN32
  HANDLE sem_;
#else
  sem_t sem_;
#endif
};

}  // namespace runtime
}  // namespace dgl

#endif  // DGL_RUNTIME_SEMAPHORE_WRAPPER_H_
