// merge.cpp

#include "volsort.h"

#include <iostream>

// Prototypes

Node *msort(Node *head, bool numeric);
void  split(Node *head, Node *&left, Node *&right);
Node *merge(Node *left, Node *right, bool numeric);

// Implementations

void merge_sort(List &l, bool numeric) {
    l.head = msort(l.head, numeric);
}

Node *msort(Node *head, bool numeric) {
    if (head == nullptr || head->next == nullptr) {
        return head;
    }

    //divide
    Node *left;
    Node *right;
    split(head, left, right);

    //conquer
    left  = msort(left, numeric);
    right = msort(right, numeric);
    //merge back
    return merge(left, right, numeric);
}

void split(Node *head, Node *&left, Node *&right) {
    //reference to gfg merge sort page 
    Node *slow = head;
    Node *fast = head->next;
 
    //fast moves 2 nodes, slow moves 1 node
    while (fast != nullptr && fast->next != nullptr) {
        slow = slow->next;
        fast = fast->next->next;
    }
 
    // fast hit the end, so slow is the last node of the first half
    left  = head;
    right = slow->next;
    slow->next = nullptr;

}
//reference to gfg again
Node *merge(Node *left, Node *right, bool numeric) {
    // temporary starting point
    Node tempo;
    Node *tail = &tempo;
 
    while (left != nullptr && right != nullptr) {
        // numeric decides which compare function to use
        bool rightSmaller;
        if (numeric) {
            rightSmaller = node_number_compare(right, left);
        } else {
            rightSmaller = node_string_compare(right, left);
        }
 
        if (rightSmaller) {
            tail->next = right;
            right = right->next;
        } else {
            tail->next = left;
            left = left->next;
        }
        tail = tail->next;  // move the tail
    }
 
    // One list is empty; attach the rest of the other one
    if (left != nullptr) {
        tail->next = left;
    } else {
        tail->next = right;
    }
 
    return tempo.next; 
}


