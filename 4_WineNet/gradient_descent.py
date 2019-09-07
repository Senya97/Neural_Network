from numpy.random import permutation


def gradient_descent(net, X_train, y_train, X_test, y_test, batch_size, num_epoch, loss_function, optimizer):
    """
    Реализация градиентного спуска.
    :param net:
    :param X_train:
    :param y_train:
    :param batch_size:
    :param num_epoch:
    :param loss_function:
    :param optimizer:
    :return:
    """
    train_accuracy_history= []
    train_loss_history = []
    test_accuracy_history = []
    test_loss_history = []
    for epoch in range(num_epoch):
        # перемешиваем обучающую выборку (индексы объектов)
        order = permutation(len(X_train))
        # делим выборку на "батчи"
        for start_index in range(0, len(X_train), batch_size):
            # обнуляем градиенты
            optimizer.zero_grad()
            # берем batch из обучающей выборки
            batch_indexes = order[start_index: start_index + batch_size]
            x_batch = X_train[batch_indexes]
            y_batch = y_train[batch_indexes]
            # forward pass
            preds = net.forward(x_batch)
            # backward pass
            loss_value = loss_function(preds, y_batch)
            loss_value.backward()
            # градиентный шаг
            optimizer.step()
        # запомним результаты на трейне
        train_preds = net.forward(X_train)
        train_accuracy_history.append((train_preds.argmax(dim=1) == y_train).float().mean())
        train_loss = loss_function(train_preds, y_train)
        train_loss_history.append(train_loss.data)
        # запомним результаты на тесте
        test_preds = net.forward(X_test)
        test_accuracy_history.append((test_preds.argmax(dim=1) == y_test).float().mean())
        test_loss = loss_function(test_preds, y_test)
        test_loss_history.append(test_loss.data)
    return train_accuracy_history, train_loss_history, test_accuracy_history, test_loss_history
