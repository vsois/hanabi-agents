#include <cmath>
#include <memory>
#include <mutex>
#include <vector>
#include <unordered_map>
// #include <iostream>

template<typename>
class SumTreeTestProxy;

template<typename Precision>
struct SumTreeNode {
  std::shared_ptr<SumTreeNode> left;
  std::shared_ptr<SumTreeNode> right;
  std::shared_ptr<SumTreeNode> parent;
  long long index = -1;
  Precision value{0.0};
  std::mutex value_mutex;
};


template<typename Precision>
class SumTree {
public:
  SumTree(size_t capacity)
  {
    root_->index = -2;
    depth_ = static_cast<size_t>(std::ceil(std::log2(capacity)));
    populateTree(root_, 0);
  }

  void updateValue(const int index, const Precision value) {
    const Precision diff = value - leaves_[index]->value;
    updateValue_(leaves_[index], diff);
  }


  void updateValues(const std::vector<size_t> indices,
                    const std::vector<Precision> values) {
    #pragma omp parallel for
    for (size_t idx = 0; idx < indices.size(); idx++) {
      updateValue(indices[idx], values[idx]);
    }
  }

  size_t getIndex(const Precision quantile) const {
    const Precision query_value = quantile * root_->value;
    // std::cout << query_value << std::endl;
    return getIndex_(root_, query_value);
  }

  std::vector<size_t> getIndices(const std::vector<Precision> query_values) const {
    std::vector<size_t> indices(query_values.size());
    #pragma omp parallel for
    for (size_t idx = 0; idx < query_values.size(); idx++) {
      indices[idx] = getIndex(query_values[idx]);
    }
    return indices;
  }

  Precision getValue(const size_t idx) const {
    return leaves_.at(idx)->value;
  }

  std::vector<Precision> getValues(const std::vector<size_t> indices) const {
    std::vector<Precision> values(indices.size());
    #pragma omp parallel for
    for (size_t i = 0; i < indices.size(); i++) {
      values[i] = getValue(indices[i]);
    }
    return values;
  }

  Precision getTotalVal() const { return root_->value; }

  size_t getCapacity() const { return leaves_.size(); }

private:

  void updateValue_(std::shared_ptr<SumTreeNode<Precision>> node,
                    const Precision diff) {
    {
      const std::lock_guard<std::mutex> lock(node->value_mutex);
      node->value += diff;
    }
    if (node->index == -2) {
      return;
    }
    updateValue_(node->parent, diff);
  }

  size_t getIndex_(std::shared_ptr<SumTreeNode<Precision>> node,
                   Precision query_value) const {
    // std::cout << "querying node " << node->value << " with value " << query_value << std::endl;
    if (node->index > -1) {
      // std::cout << "reached leaf " << node->index << " with value " << node->value << std::endl;
      return node->index;
    }
    if (query_value < node->left->value) {
      // std::cout << "going left" << std::endl;
      return getIndex_(node->left, query_value);
    }
    // std::cout << "going right" << std::endl;
    return getIndex_(node->right, query_value - node->left->value);
  }

  void populateTree(std::shared_ptr<SumTreeNode<Precision>> node,
                    size_t current_depth) {
    if (current_depth == depth_) {
      node->index = leaves_.size();
      // leaves_.push_back(node);
      leaves_.insert({node->index, node});
      return;
    }
    node->left = std::shared_ptr<SumTreeNode<Precision>>(
        new SumTreeNode<Precision>());
    node->left->parent = node;
    node->right = std::shared_ptr<SumTreeNode<Precision>>(
        new SumTreeNode<Precision>());
    node->right->parent = node;
    populateTree(node->left, current_depth + 1);
    populateTree(node->right, current_depth + 1);
  }

  size_t depth_{0};
  std::shared_ptr<SumTreeNode<Precision>> root_{new SumTreeNode<Precision>()};
  std::unordered_map<size_t, std::shared_ptr<SumTreeNode<Precision>>> leaves_;

  friend SumTreeTestProxy<Precision>;
};
