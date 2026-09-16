// stl.cpp

#include "volsort.h"

#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;
void stl_sort(List &l, bool numeric) {
    if (l.head == nullptr) return;
    //copying node pointers into a vectors
    vector<Node*> nodes;
    for (Node *curr = l.head; curr != nullptr; curr = curr->next) {
        nodes.push_back(curr);
    }
    //sort them
    if (numeric) {
        sort(nodes.begin(), nodes.end(), node_number_compare);
    } else {
        sort(nodes.begin(), nodes.end(), node_string_compare);
    }

    //relink node in sorted order
    for (size_t i = 0; i + 1 < nodes.size(); i++) {
        nodes[i]->next = nodes[i + 1];
    }
    nodes[nodes.size() - 1]->next = nullptr;

    //update head
    l.head = nodes[0];
}

