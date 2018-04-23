import numpy as np
from rl2048.utils import action_select, copy_data, variable


def test_action_select():

    x = variable((100, 4), type_='float')
    a = variable((100, ), type_='long')

    x_data = np.random.random(size=(100, 4))
    a_data = np.random.randint(0, 4, size=100)

    copy_data(a, a_data)
    copy_data(x, x_data)

    expected = x_data[range(100), a_data]
    computed = action_select(x, a).data.cpu().numpy()

    np.testing.assert_allclose(expected, computed)
