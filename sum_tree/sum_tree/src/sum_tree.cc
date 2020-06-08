#include "sum_tree.h"

SumTree::SumTree(size_t capacity)
  : capacity_(capacity)
{
  depth_ = static_cast<size_t>(std::ceil(std::log2(capacity)));
  populateTree(root_, 0);
}

void SumTree::updateValue(const int index, const double value) {
  const double diff = leaves_[index]->value - value;
  updateValue_(leaves_[index], diff);
}

void SumTree::updateValues(const std::vector<size_t> indices,
                  const std::vector<double> values) {
  for (size_t idx = 0; idx < indices.size(); idx++) {
    updateValue(indices[idx], values[idx]);
  }
}
  
size_t SumTree::getIndex(const double query_value) {
  return getIndex_(root_, query_value);
}

std::vector<size_t> SumTree::getIndices(
    const std::vector<double> query_values) {
  std::vector<size_t> indices(query_values.size());
  for (size_t idx = 0; idx < query_values.size(); idx++) {
    indices[idx] = getIndex(query_values[idx]);
  }
  return indices;
}

void SumTree::updateValue_(std::shared_ptr<SumTreeNode> node,
                           const double value) {
  if (!node->parent) {
    return;
  }
  {
    const std::lock_guard<std::mutex> lock(node->value_mutex);
    node->value += value;
  }
  updateValue_(node->parent, value);
}

size_t SumTree::getIndex_(std::shared_ptr<SumTreeNode> node,
                          double query_value) {
  if (!node->left) {
    return node->index;
  }
  if (query_value < node->left->value) {
    return getIndex_(node->left, query_value);
  }
  return getIndex_(node->right, query_value - node->left->value);
}

void SumTree::populateTree(std::shared_ptr<SumTreeNode> node,
                           size_t current_depth) {
  if (current_depth == depth_) {
    node->index = leaves_.size();
    leaves_.push_back(node);
    return;
  }
  node->left = std::shared_ptr<SumTreeNode>(new SumTreeNode());
  node->left->parent = node;
  node->right = std::shared_ptr<SumTreeNode>(new SumTreeNode());
  node->right->parent = node;
  populateTree(node->left, current_depth + 1);
  populateTree(node->right, current_depth + 1);
}
