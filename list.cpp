#include "volsort.h"

using namespace std;
List::List() {
    head = nullptr;
    size = 0;
}

List::~List() {
    Node *curr = head;
    while (curr != nullptr) {
        Node *next = curr->next;   // save next BEFORE deleting curr
        delete curr;
        curr = next;
    }
}

void List::push_front(const std::string &s) {
    Node *n = new Node;
    n->string = s;
 
    // Convert the string to a number; use 0 if it isn't a number
    if (isdigit(s[0]) || s[0] == '-') n->number = std::stoi(s);
    else n->number = 0;
 
    n->next = head;
    head = n;
    size++;
}

bool node_number_compare(const Node *a, const Node *b) {
    return a->number < b->number;
}
 
bool node_string_compare(const Node *a, const Node *b) {
    return a->string < b->string;
}