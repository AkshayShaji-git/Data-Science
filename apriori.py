from itertools import combinations
from collections import defaultdict

class Apriori:
    def __init__(self, min_support=0.5, min_confidence=0.7):
        """
        Initializes the Apriori algorithm.
        """
        self.min_support = min_support
        self.min_confidence = min_confidence

    def fit(self, transactions):
        """
        Runs the Apriori algorithm on a list of transactions.

        """
        self.transactions = transactions
        self.itemsets, self.support_data = self._generate_frequent_itemsets()
        self.rules = self._generate_association_rules()

    def _generate_frequent_itemsets(self):
        """
        Generates frequent itemsets using the Apriori principle.

        """
        item_counts = defaultdict(int)
        transaction_count = len(self.transactions)

        # Count frequency of individual items
        for transaction in self.transactions:
            for item in transaction:
                item_counts[frozenset([item])] += 1

        # Get frequent 1-itemsets
        current_itemsets = {itemset for itemset, count in item_counts.items() 
                            if count / transaction_count >= self.min_support}

        frequent_itemsets = list(current_itemsets)
        support_data = {itemset: count / transaction_count for itemset, count in item_counts.items()}

        k = 2  # Start with pairs
        while current_itemsets:
            candidate_itemsets = self._generate_candidates(current_itemsets, k)
            item_counts = defaultdict(int)

            for transaction in self.transactions:
                transaction_set = frozenset(transaction)
                for candidate in candidate_itemsets:
                    if candidate.issubset(transaction_set):
                        item_counts[candidate] += 1

            current_itemsets = {itemset for itemset, count in item_counts.items() 
                                if count / transaction_count >= self.min_support}

            support_data.update({itemset: count / transaction_count for itemset, count in item_counts.items()})
            frequent_itemsets.extend(current_itemsets)
            k += 1

        return frequent_itemsets, support_data

    def _generate_candidates(self, prev_itemsets, k):
        """
        Generates candidate k-itemsets from (k-1)-itemsets.
        """
        candidate_itemsets = set()
        prev_itemsets_list = list(prev_itemsets)

        for i in range(len(prev_itemsets_list)):
            for j in range(i + 1, len(prev_itemsets_list)):
                itemset1 = prev_itemsets_list[i]
                itemset2 = prev_itemsets_list[j]

                union_itemset = itemset1 | itemset2
                if len(union_itemset) == k:
                    candidate_itemsets.add(union_itemset)

        return candidate_itemsets

    def _generate_association_rules(self):
        """
        Generates association rules from frequent itemsets.
        """
        rules = []

        for itemset in self.itemsets:
            if len(itemset) > 1:
                for i in range(1, len(itemset)):
                    for subset in combinations(itemset, i):
                        antecedent = frozenset(subset)
                        consequent = itemset - antecedent

                        support_antecedent = self.support_data.get(antecedent, 0)
                        support_itemset = self.support_data[itemset]

                        confidence = support_itemset / support_antecedent if support_antecedent > 0 else 0
                        if confidence >= self.min_confidence:
                            rules.append((antecedent, consequent, confidence))

        return rules

    def print_rules(self):
        """
        Prints the association rules.
        """
        print("\nAssociation Rules:")
        for antecedent, consequent, confidence in self.rules:
            print(f"{set(antecedent)} → {set(consequent)} (Confidence: {confidence:.2f})")

# Example Usage:
transactions = [
    ['milk', 'bread', 'butter'],
    ['beer', 'bread'],
    ['milk', 'bread', 'butter'],
    ['beer', 'butter'],
    ['milk', 'butter'],
    ['bread', 'butter']
]

apriori = Apriori(min_support=0.4, min_confidence=0.6)
apriori.fit(transactions)
apriori.print_rules()
