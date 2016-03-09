#Preview of the implicit surfaces
```
1+sin(x)*cos(y)-1/sqrt(z)
```
- Mesh generated in 0.47s (174556 triangles) (128x128x128 grid)
- KDTree built in 7.45s (average depth 20.4)
- Render took 138.25s

![1](images/1.jpg?raw=true)
![6](images/6.jpg?raw=true)

```
x%2+y%3+z*z-1+cos(x)*sin(y)*2 
```
- Mesh generated in 0.73s (131844 triangles) (128x128x128 grid)
- KDTree built in 7.62s (average depth 23.6)
- Render took 32.98s

![2](images/2.jpg?raw=true)

```
(5-sqrt(x*x+y*y))^2+z^2-4
```
- Mesh generated in 0.44s (71872 triangles) (128x128x128 grid)
- KDTree built in 3.65s (average depth 21.1)
- Render took 9.13s

![3](images/3.jpg?raw=true)

```
(5-sqrt(x*x+y*y))^2+z^2-4
```
- Mesh generated in 0.74s (21464 triangles) (128x128x128 grid)
- KDTree built in 1.68s (average depth 19.2)
- Render took 2.81s

![4](images/4.jpg?raw=true)

```
x^4+y^4+z^4-1.5*(x^2+y^2+z^2)+1
```
- Mesh generated in 0.36s (102112 triangles) (128x128x128)
- KDTree built in 6.87s (average depth 29.7)
- Render took 17.12s

![5](images/5.jpg?raw=true)

```
1+cos(x)+cos(y)+cos(z)
```
- Mesh generated in 0.26s (11388 triangles) (32x32x32)
- KDTree built in 0.60s (average depth 13.9)
- Render took 59.82s
![7](images/7.jpg?raw=true)

```
sin(x)+sin(y)+sin(z)
```
- Mesh generated in 0.30s (20034 triangles) (32x32x32)
- KDTree built in 1.03s (average depth 14.7)
- Render took 87.02s
![8](images/8.jpg?raw=true)

```
cos(x)+cos(y)+z
```
- Mesh generated in 0.29s (4338 triangles) (31x31x31)
- KDTree built in 0.50s (average depth 11.5)
- Render took 17.41s
![9](images/9.jpg?raw=true)

```
cos(cos(x)+cos(y))+z
```
- Mesh generated in 0.18s (2930 triangles) (31x31x31)
- KDTree built in 0.19s (average depth 10.3)
- Render took 10.08s
![10](images/10.jpg?raw=true)

```
sqrt(sqrt((cos(x)+cos(y))^2))+z
```
> (When you haven't implemented abs function)

- Mesh generated in 0.28s (3446 triangles) (31x31x31)
- KDTree built in 0.28s (average depth 11.0)
- Render took 8.89s
![11](images/11.jpg?raw=true)




