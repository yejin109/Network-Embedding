import networkx as nx
import matplotlib.pyplot as plt


# Graph
G = nx.Graph()
for index, case in enumerate(adj_mat_ind):
    if index % 2 != 0:
        continue
    G.add_edge(case[0], case[1])

pos = nx.spring_layout(G)
plt.figure(figsize=(16, 12))

plt.subplot(2, 1, 1)
nx.draw(G, pos, with_labels=True, node_color='#5c6bf7')
nx.draw_networkx_nodes(G, pos, nodelist=list(np.where(label == 1)[0]), node_color='r')
plt.title('Target')

plt.subplot(2, 1, 2)
plt.scatter(model.snd_target[is_one, 0], model.snd_target[is_one, 1], c='red')
plt.scatter(model.snd_target[~is_one, 0], model.snd_target[~is_one, 1], c='blue')
for index in range(nodes_no):
    plt.annotate(f'{index}', (model.snd_target[index, 0], model.snd_target[index, 1]))
plt.title(f'Embedding for 2nd Proximity')
plt.show()

plt.figure()
plt.plot(loss_per_epoch_snd, linewidth=1)
plt.title('LPE for 2nd')
plt.show()


# t-SNE
label = np.loadtxt('../data_set/karate_label.txt')
is_one = label[:, 1] == 1

case = input('케이스 이름 알려주기')
perp = 4
setup['d'] = embedding_size
setup['n'] = learn_rate
setup['t'] = walk_length
setup['w'] = window_size
setup['r'] = walks_per_vertex
setup['perp'] = perp
with open(f'setup/{case}.json', 'w', encoding='UTF-8-sig') as fp:
    fp.write(json.dumps(setup, ensure_ascii=False))

transforms = dict()

model_tsne_5 = TSNE(random_state=0, perplexity=perp)
transformed_5 = model_tsne_5.fit_transform(model_5.first_phi)
transforms[5] = transformed_5

model_tsne_8 = TSNE(random_state=0, perplexity=perp)
transformed_8 = model_tsne_5.fit_transform(model_8.first_phi)
transforms[8] = transformed_8

model_tsne_10 = TSNE(random_state=0, perplexity=perp)
transformed_10 = model_tsne_5.fit_transform(model_10.first_phi)
transforms[10] = transformed_10

model_tsne_12 = TSNE(random_state=0, perplexity=perp)
transformed_12 = model_tsne_5.fit_transform(model_12.first_phi)
transforms[12] = transformed_12

plt.figure()
test_set = [5 ,8, 10, 12]

for index in range(4):
    plt.subplot(2, 2, index+1)
    plot = transforms[test_set[index]]
    plt.scatter(plot[is_one, 0], plot[is_one, 1], c='red')
    plt.scatter(plot[~is_one, 0], plot[~is_one, 1], c='blue')
    plt.title(f'Perplexity : {perp}, Window Size :{test_set[index]}')

plt.savefig(f'D:/Yejin/From now on,/CSE URP/URP/Week 4/Figure/case {case} win size.png')
plt.show()