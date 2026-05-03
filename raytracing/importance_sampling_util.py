def balance_heuristic(n_a, pdf_a, n_b, pdf_b):
    prod_a = n_a * pdf_a
    return prod_a / (prod_a + n_b * pdf_b)


def power_2_heuristic(n_a, pdf_a, n_b, pdf_b):
    prod_a = n_a * pdf_a
    prod_b = n_b * pdf_b
    power_a = prod_a * prod_a
    return power_a / (power_a + prod_b * prod_b)
