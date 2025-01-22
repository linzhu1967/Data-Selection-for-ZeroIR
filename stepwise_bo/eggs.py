def distribute_eggs(eggs, baskets):
    if baskets == 1:
        return [[eggs]]
    else:
        distributions = []
        for eggs_in_basket in range(eggs + 1):
            for distribution in distribute_eggs(eggs - eggs_in_basket, baskets - 1):
                distributions.append([eggs_in_basket] + distribution)
        return distributions