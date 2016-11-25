from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

dataset = pd.read_csv('data_all_without_health.csv')
print(dataset)

KM = KMeans(n_clusters=2, init='k-means++', random_state=170)

KM = KM.fit(dataset)

print("The cluster centroids are: \n", KM.cluster_centers_)
print("Cluster", KM.labels_)
print("Sum of distances of samples to their closest cluster center: ", KM.inertia_)

colors = ['blue','yellow']

def Tempo(y_data, y_name):
    plt.scatter(dataset.tempo, y_data, c=KM.labels_, cmap=matplotlib.colors.ListedColormap(colors), s=75)
    plt.xlabel('tempo')
    plt.ylabel(y_name)
    plt.show()
Tempo(dataset.popularity, 'popularity');
Tempo(dataset.valence, 'valence');
Tempo(dataset.energy, 'energy');
Tempo(dataset.dance, 'dance');
Tempo(dataset.acoustic, 'acoustic');
Tempo(dataset.instrumental, 'instrumental');
# #
def Popularity(y_data, y_name):
    plt.scatter(dataset.popularity, y_data, c=KM.labels_, cmap=matplotlib.colors.ListedColormap(colors), s=75)
    plt.xlabel('popularity')
    plt.ylabel(y_name)
    plt.show()
Popularity(dataset.valence, 'valence');
Popularity(dataset.energy, 'energy');
Popularity(dataset.dance, 'dance');
Popularity(dataset.acoustic, 'acoustic');
Popularity(dataset.instrumental, 'instrumental');

def Valence(y_data, y_name):
    plt.scatter(dataset.valence, y_data, c=KM.labels_, cmap=matplotlib.colors.ListedColormap(colors), s=75)
    plt.xlabel('valence')
    plt.ylabel(y_name)
    plt.show()
Valence(dataset.energy, 'energy');
Valence(dataset.dance, 'dance');
Valence(dataset.acoustic, 'acoustic');
Valence(dataset.instrumental, 'instrumental');
#
def Energy(y_data, y_name):
    plt.scatter(dataset.energy, y_data, c=KM.labels_, cmap=matplotlib.colors.ListedColormap(colors), s=75)
    plt.xlabel('energy')
    plt.ylabel(y_name)
    plt.show()
Energy(dataset.dance, 'dance');
Energy(dataset.acoustic, 'acoustic');
Energy(dataset.instrumental, 'instrumental');

def Dance(y_data, y_name):
    plt.scatter(dataset.dance, y_data, c=KM.labels_, cmap=matplotlib.colors.ListedColormap(colors), s=75)
    plt.xlabel('dance')
    plt.ylabel(y_name)
    plt.show()
Dance(dataset.acoustic, 'acoustic');
Dance(dataset.instrumental, 'instrumental');

def Acoustic(y_data, y_name):
    plt.scatter(dataset.acoustic, y_data, c=KM.labels_, cmap=matplotlib.colors.ListedColormap(colors), s=75)
    plt.xlabel('acoustic')
    plt.ylabel(y_name)
    plt.show()
Acoustic(dataset.instrumental, 'instrumental');

# def Tempo3D(x_name, x_data, y_name, y_data):
#     fig = plt.figure(1, figsize=(8, 6))
#     ax = Axes3D(fig, elev=-150, azim=110)
#     ax.scatter(x_data, y_data,dataset.tempo,  c=KM.labels_, s=75)
#     plt.xlabel(x_name)
#     plt.ylabel(y_name)
#     plt.show()
# Tempo3D('popularity', dataset.popularity, 'valence', dataset.valence);
# Tempo3D('popularity', dataset.popularity, 'energy', dataset.energy);
# Tempo3D('popularity', dataset.popularity, 'dance', dataset.dance);
# Tempo3D('popularity', dataset.popularity, 'acoustic', dataset.acoustic);
# Tempo3D('popularity', dataset.popularity, 'instrumental', dataset.instrumental);
# Tempo3D('valence', dataset.valence, 'energy', dataset.energy);
# Tempo3D('valence', dataset.valence, 'dance', dataset.dance);
# Tempo3D('valence', dataset.valence, 'acoustic', dataset.acoustic);
# Tempo3D('valence', dataset.valence, 'instrumental', dataset.instrumental);
# Tempo3D('energy', dataset.energy, 'dance', dataset.dance);
# Tempo3D('energy', dataset.energy, 'acoustic', dataset.acoustic);
# Tempo3D('energy', dataset.energy, 'instrumental', dataset.instrumental);
# Tempo3D('dance', dataset.dance, 'acoustic', dataset.acoustic);
# Tempo3D('dance', dataset.dance, 'instrumental', dataset.instrumental);
# Tempo3D('acoustic', dataset.acoustic, 'instrumental', dataset.instrumental);
#
# def Popularity3D(x_name, x_data, y_name, y_data):
#     fig = plt.figure(1, figsize=(8, 6))
#     ax = Axes3D(fig, elev=-150, azim=110)
#     ax.scatter(x_data, y_data,dataset.popularity,  c=KM.labels_, s=75)
#     plt.xlabel(x_name)
#     plt.ylabel(y_name)
#     plt.show()
# Popularity3D('valence', dataset.valence, 'energy', dataset.energy);
# Popularity3D('valence', dataset.valence, 'dance', dataset.dance);
# Popularity3D('valence', dataset.valence, 'acoustic', dataset.acoustic);
# Popularity3D('valence', dataset.valence, 'instrumental', dataset.instrumental);
# Popularity3D('energy', dataset.energy, 'dance', dataset.dance);
# Popularity3D('energy', dataset.energy, 'acoustic', dataset.acoustic);
# Popularity3D('energy', dataset.energy, 'instrumental', dataset.instrumental);
# Popularity3D('dance', dataset.dance, 'acoustic', dataset.acoustic);
# Popularity3D('dance', dataset.dance, 'instrumental', dataset.instrumental);
# Popularity3D('acoustic', dataset.acoustic, 'instrumental', dataset.instrumental);
#
# def Valence3D(x_name, x_data, y_name, y_data):
#     fig = plt.figure(1, figsize=(8, 6))
#     ax = Axes3D(fig, elev=-150, azim=110)
#     ax.scatter(x_data, y_data,dataset.valence,  c=KM.labels_, s=75)
#     plt.xlabel(x_name)
#     plt.ylabel(y_name)
#     plt.show()
# Valence3D('energy', dataset.energy, 'dance', dataset.dance);
# Valence3D('energy', dataset.energy, 'acoustic', dataset.acoustic);
# Valence3D('energy', dataset.energy, 'instrumental', dataset.instrumental);
# Valence3D('dance', dataset.dance, 'acoustic', dataset.acoustic);
# Valence3D('dance', dataset.dance, 'instrumental', dataset.instrumental);
# Valence3D('acoustic', dataset.acoustic, 'instrumental', dataset.instrumental);
#
# def Energy3D(x_name, x_data, y_name, y_data):
#     fig = plt.figure(1, figsize=(8, 6))
#     ax = Axes3D(fig, elev=-150, azim=110)
#     ax.scatter(x_data, y_data,dataset.energy,  c=KM.labels_, s=75)
#     plt.xlabel(x_name)
#     plt.ylabel(y_name)
#     plt.show()
# Energy3D('dance', dataset.dance, 'acoustic', dataset.acoustic);
# Energy3D('dance', dataset.dance, 'instrumental', dataset.instrumental);
# Energy3D('acoustic', dataset.acoustic, 'instrumental', dataset.instrumental);
#
# def Dance3D(x_name, x_data, y_name, y_data):
#     fig = plt.figure(1, figsize=(8, 6))
#     ax = Axes3D(fig, elev=-150, azim=110)
#     ax.scatter(x_data, y_data,dataset.dance,  c=KM.labels_, s=75)
#     plt.xlabel(x_name)
#     plt.ylabel(y_name)
#     plt.show()
# Dance3D('acoustic', dataset.acoustic, 'instrumental', dataset.instrumental);







# fig = plt.figure(1, figsize=(8, 6))
# ax = Axes3D(fig, elev=-150, azim=110)
# ax.scatter(dataset.tempo, dataset.valence, dataset.health, c=KM.labels_, s=75)
# plt.xlabel('tempo')
# plt.ylabel('valence')
# plt.show()

# plt.scatter(dataset.tempo, dataset.valence, c=KM.labels_, cmap=matplotlib.colors.ListedColormap(colors), s=75)
# plt.xlabel('tempo')
# plt.ylabel('valence')
# plt.show()
#
# plt.scatter(dataset.tempo, dataset.energy, c=KM.labels_, cmap=matplotlib.colors.ListedColormap(colors), s=75)
# plt.xlabel('tempo')
# plt.ylabel('energy')
# plt.show()
#
# plt.scatter(dataset.tempo, dataset.dance, c=KM.labels_, cmap=matplotlib.colors.ListedColormap(colors), s=75)
# plt.xlabel('tempo')
# plt.ylabel('dance')
# plt.show()
#
# plt.scatter(dataset.tempo, dataset.acoustic, c=KM.labels_, cmap=matplotlib.colors.ListedColormap(colors), s=75)
# plt.xlabel('tempo')
# plt.ylabel('acoustic')
# plt.show()
#
# plt.scatter(dataset.tempo, dataset.instrumental, c=KM.labels_, cmap=matplotlib.colors.ListedColormap(colors), s=75)
# plt.xlabel('tempo')
# plt.ylabel('instrumental')
# plt.show()