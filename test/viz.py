N = int(meta['HEIGHT']/meta['SPACING'])
x = np.linspace(-meta['HEIGHT'],meta['HEIGHT'],N)

X,Y = np.meshgrid(x,x)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.contourf(Y,X,I[:,:,0], zdir='z', offset=0, cmap=cm.gray)

for C in cylinders:
    d = C[0]
    c = C[1]
    r = C[2]
    h = C[3]
    p = C[4]
    in_ = C[5]

    Ps = geometry.cylinder_surface(-h/2, h/2, 5, c,d,r)

    ax.plot_surface(Ps[:,:,0], Ps[:,:,1], Ps[:,:,2], color='b')

ax.set_xlim(-meta['HEIGHT'],meta['HEIGHT'])
ax.set_ylim(-meta['HEIGHT'],meta['HEIGHT'])
ax.set_zlim(-meta['HEIGHT'],meta['HEIGHT'])
plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.contourf(Y,X,I[:,:,0], zdir='z', offset=0, cmap=cm.gray)

for C in cylinders:
    d = C[0]
    c = C[1]
    r = C[2]
    h = C[3]
    p = C[4]
    in_ = C[5]

    if len(in_)>0:
        ax.scatter(in_[:,0],in_[:,1],in_[:,2],color='g')
#
# ax.scatter(path_points[:,0],path_points[:,1],path_points[:,2],
#     color='k')
#
ax.set_xlim(-meta['HEIGHT'],meta['HEIGHT'])
ax.set_ylim(-meta['HEIGHT'],meta['HEIGHT'])
ax.set_zlim(-meta['HEIGHT'],meta['HEIGHT'])
plt.show()
plt.close()
