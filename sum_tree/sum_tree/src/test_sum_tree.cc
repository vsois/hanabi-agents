#include "sum_tree.h"
#include "gtest/gtest.h"
#include <memory>
#include <unordered_map>


template<typename Precision>
class SumTreeTestProxy {

public:
  SumTreeTestProxy(size_t capacity) : st(capacity) {
  }

  // void TearDown() override {}
  std::shared_ptr<SumTreeNode<Precision>> getRoot() {return st.root_;}
  std::unordered_map<size_t, std::shared_ptr<SumTreeNode<Precision>>>& getLeaves() {return st.leaves_;}


  SumTree<Precision> st;
};

template<typename Precision>
void assert_nodes_zero(std::shared_ptr<SumTreeNode<Precision>> node) {
  if (!node->left) {
    return;
  }
  assert_nodes_zero(node->left);
  assert_nodes_zero(node->right);
  ASSERT_EQ(node->value, 0.0);
  if (node->left->left) {
    ASSERT_EQ(node->index, -1);
  }
}

template<typename Precision>
void assert_parent(std::shared_ptr<SumTreeNode<Precision>> node) {
  ASSERT_TRUE(node->parent);
  if (!node->left) {
    return;
  }
  assert_parent(node->left);
  assert_parent(node->right);
}

TEST(CtorTest, HandlesTreeCreation) {
  SumTree<float> st2(7);
  ASSERT_EQ(st2.getCapacity(), 8);
  SumTree<float> st3(9);


  ASSERT_EQ(st3.getCapacity(), 16);
  SumTreeTestProxy<float> stt(8);
  ASSERT_EQ(stt.getRoot()->index, -2);
  ASSERT_EQ(stt.getRoot()->value, 0.0);
  ASSERT_EQ(stt.st.getCapacity(), 8);
  ASSERT_EQ(stt.st.getTotalVal(), 0.0);
  auto& leaves = stt.getLeaves();
  for (size_t i = 0; i < leaves.size(); i++) {
    const auto& leaf = leaves[i];
    ASSERT_EQ(leaf->value, 0.0);
    ASSERT_EQ(leaf->index, i);
  }

  assert_nodes_zero(stt.getRoot()->left);
  assert_nodes_zero(stt.getRoot()->right);
  assert_parent(stt.getRoot()->left);
  assert_parent(stt.getRoot()->right);
}


template<typename Precision>
void print_nodes(std::shared_ptr<SumTreeNode<Precision>> node) {
  std::cout << "Node: index " << node->index << ", value " << node->value << " ";
  if (!node->left) {
    return;
  }
  print_nodes(node->left);
  print_nodes(node->right);
}

TEST(UpdateValueTest, HandlesValue) {
  SumTreeTestProxy<float> stt(4);
  stt.st.updateValue(0, 1.0);
  ASSERT_EQ(stt.getLeaves()[0]->value, 1.0);
  ASSERT_EQ(stt.st.getTotalVal(), 1.0);
  stt.st.updateValue(2, 2.0);
  ASSERT_EQ(stt.getLeaves()[2]->value, 2.0);
  ASSERT_EQ(stt.st.getTotalVal(), 3.0);
  stt.st.updateValue(2, 1.0);
  ASSERT_EQ(stt.getLeaves()[2]->value, 1.0);
  ASSERT_EQ(stt.st.getTotalVal(), 2.0);
}

TEST(UpdateValuesTest, HandlesValues) {
  SumTreeTestProxy<float> stt(4);
  std::vector<float> values{1.0, 2.0, 3.0, 4.0};
  stt.st.updateValues({0, 1, 2, 3}, values);
  ASSERT_EQ(stt.st.getTotalVal(), 10.0);
  auto leaves = stt.getLeaves();
  for (size_t i = 0; i < leaves.size(); i++) {
    ASSERT_EQ(values[i], leaves[i]->value);
  }
}

TEST(GetIndexTest, HandlesIndex) {
  SumTreeTestProxy<float> stt(4);
  stt.st.updateValues({0, 1, 2, 3}, {1.0, 2.0, 3.0, 4.0});
  print_nodes(stt.getRoot());
  ASSERT_EQ(stt.st.getIndex(0.0), 0);
  ASSERT_EQ(stt.st.getIndex(0.099), 0);
  ASSERT_EQ(stt.st.getIndex(0.1), 1);
  ASSERT_EQ(stt.st.getIndex(0.299), 1);
  ASSERT_EQ(stt.st.getIndex(0.3), 2);
  ASSERT_EQ(stt.st.getIndex(0.599), 2);
  ASSERT_EQ(stt.st.getIndex(0.6), 3);
  ASSERT_EQ(stt.st.getIndex(1.0), 3);
}
