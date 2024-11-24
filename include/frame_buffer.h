#pragma once

#include <deque>

#include "data_frame.h"

template <typename E>
class FrameBuffer : public std::deque<E> {
 public:
  explicit FrameBuffer(int size);
  ~FrameBuffer() = default;

  void append(E e);
  void prepend(E e);

 private:
  int size_;
};

template <typename E>
FrameBuffer<E>::FrameBuffer(int size) : size_(size) {}

template <typename E>
void FrameBuffer<E>::append(E e) {
  this->push_front(e);
  if (static_cast<int>(this->size()) > size_) {
    this->pop_back();
  }
}

template <typename E>
void FrameBuffer<E>::prepend(E e) {
  this->push_back(e);
  if (static_cast<int>(this->size()) > size_) {
    this->pop_front();
  }
}
