import copy
from bitarray import bitarray


def get_pairs(hand):
    result_pairs = []
    result_hands = []
    for i in range(len(hand)):
        if hand[i] >= 2:
            new_pair = [(i % 9) + 1] * 2
            result_pairs.append(new_pair)

            copy_hand = hand.copy()
            copy_hand[i] -= 2
            result_hands.append(copy_hand)
    return result_pairs, result_hands


def get_groups(hand):
    result_groups = []
    result_hands = []
    # mod 3 needs to equal 0
    for i in range(3):
        if 0 < sum(hand[i * 9:(i + 1) * 9]) <= 2:
            return [], []

        for j in range(9):
            index = 9 * i + j

            if hand[index] >= 3:
                new_triplet = [j + 1] * 3
                result_groups.append(new_triplet)

                copy_hand = hand.copy()
                copy_hand[index] -= 3
                result_hands.append(copy_hand)

            if j <= 6 and hand[index] >= 1 and hand[index + 1] >= 1 and hand[index + 2] >= 1:
                new_sequence = [j + 1, j + 2, j + 3]
                result_groups.append(new_sequence)

                copy_hand = hand.copy()
                copy_hand[index] -= 1
                copy_hand[index + 1] -= 1
                copy_hand[index + 2] -= 1
                result_hands.append(copy_hand)

    for i in range(7):
        index = i + 27
        if hand[index] >= 3:
            new_triplet = [i + 1] * 3
            result_groups.append(new_triplet)

            copy_hand = hand.copy()
            copy_hand[index] -= 3
            result_hands.append(copy_hand)

    return result_groups, result_hands


def recurse_groups(hand):
    # print(convert_back(hand))

    if sum(hand) == 0:
        return [], []

    result_group_set = []
    test_groups, test_hands = get_groups(hand)
    for i in range(len(test_hands)):
        test_group_sets = recurse_groups(test_hands[i])
        for test_group_set in test_group_sets:
            test_group_set.append(test_groups[i])
            result_group_set.append(test_group_set)

    return result_group_set


def check_winning_hand(hand):
    winning_group_sets = []

    test_pairs, test_hands = get_pairs(hand)
    for i in range(len(test_hands)):
        winning_groups = recurse_groups(test_hands[i])
        for winning_group in winning_groups:
            winning_group.append(test_pairs[i])
            winning_group_sets.append(winning_group)
    return winning_group_sets


def convert(hand):
    result = [0] * 34
    for i in range(len(hand)):
        for tile in hand[i]:
            result[i*9 + tile - 1] += 1
    return result


def convert_back(hand):
    result = [[], [], [], []]
    for i in range(len(hand)):
        for j in range(hand[i]):
            result[i // 9].append((i % 9) + 1)
    return result


test_hand = [[3, 4, 5, 5, 6, 7], [5, 5, 6, 7, 8], [6, 7, 8], []]
print(test_hand)
print()
test_hand = convert(test_hand)

# p, h = get_pairs(test_hand)
# g, h = get_groups(test_hand)

# for index_variable in range(len(g)):
#     print("Group: ")
#     print(g[index_variable])
#     print("Hand: ")
#     print(convert_back(h[index_variable]))
#     print("\n")
result_group_sets = check_winning_hand(test_hand)
if not result_group_sets:
    print("Winning group set is empty")
else:
    for index_variable in range(len(result_group_sets)):
        print("Winning group set " + str(index_variable) + ": ")
        print(result_group_sets[index_variable])
        print()
