import numpy as np
import matplotlib.pyplot as plt

# Constanten in SI-eenheden, VOORLOPIG
m = 0.4          #massa robot
g = 9.81          #zwaarteveldversnelling
C_r = 0.03     #rolweerstandscoëfficiënt
C_D = 1.15         #luchtweerstandscoëfficiënt
rho = 1.2        #massadichtheid lucht
A = 0.02           #frontale oppervlakte robot
eta = 0.8         #rendement motor
P = 1.5          # ingaand vermogen
mu = 0.4        # statische wrijvingscoëfficiënt
F_stat_friction = m * g * mu  # statische wrijvingskracht

# Baan: y(x) en y'(x), bepaald eerder in dit verslag
def y(x):
    return pow(x, 2)/250 -x/25

def dy_dx(x):#afgeleide van y(x)
    return x/125 - 1/25

# tangentiële versnelling, de normale versnelling is niet relevant
def acceleration(x, v):
    yprime = dy_dx(x)
    slope_factor = (-yprime - C_r) / np.sqrt(1 + yprime**2) #eerste 2 termen van de DV na OIF
    F_grav_friction = m * g * slope_factor #zwaartekracht en rolweerstand
    F_drag = 0.5 * C_D * rho * A * v**2 #luchtweerstand
    F_max = 10  # maximum motorkracht
    F_prop = min(eta * P / max(v, 0.001), F_max) #beperk de aandrijfkracht tot 10N, vermijd nuldeling
    F_t = F_grav_friction - F_drag + F_prop #tangentiële resulterende kracht

    if v==0:
        return F_t / m if F_t > F_stat_friction else 0 #tangentiële versnelling (2e wet Newton of 0)
    else:
        return F_t / m  # tangentiële versnelling (2e wet Newton)

# tijdsafgeleide van x, bepaald d.m.v. kettingregel, zie eerder in verslag
def dx_dt(x, v):
    yprime = dy_dx(x)
    return v / np.sqrt(1 + yprime**2)

# Runge-Kutta stap (voor s, v, x)
def rk4_step(s, v, x, dt):#methode om een differentiaalvergelijking benaderend op te lossen
    a1 = acceleration(x, v)
    dx1 = dx_dt(x, v)

    s1 = v
    v1 = a1
    x1 = dx1

    a2 = acceleration(x + 0.5*dt*x1, v + 0.5*dt*v1)
    dx2 = dx_dt(x + 0.5*dt*x1, v + 0.5*dt*v1)
    s2 = v + 0.5*dt*v1
    v2 = a2
    x2 = dx2

    a3 = acceleration(x + 0.5*dt*x2, v + 0.5*dt*v2)
    dx3 = dx_dt(x + 0.5*dt*x2, v + 0.5*dt*v2)
    s3 = v + 0.5*dt*v2
    v3 = a3
    x3 = dx3

    a4 = acceleration(x + dt*x3, v + dt*v3)
    dx4 = dx_dt(x + dt*x3, v + dt*v3)
    s4 = v + dt*v3
    v4 = a4
    x4 = dx4

    s_next = s + (dt/6)*(s1 + 2*s2 + 2*s3 + s4)
    v_next = v + (dt/6)*(v1 + 2*v2 + 2*v3 + v4)
    x_next = x + (dt/6)*(x1 + 2*x2 + 2*x3 + x4)

    return s_next, v_next, x_next

dt = 0.001 #om de hoeveel seconden er een waarde wordt berekend, hoe kleiner hoe nauwkeuriger
t_max = 10 #tot wanneer de waarden worden berekend
n_steps = int(t_max / dt) #aantal stappen die er bijgevolg gezet moeten worden

# Beginwaarden
s = 0.0
v = 0.0
x = 0.0

t_vals = [0]
s_vals = [s]
v_vals = [v]
a_vals = [acceleration(x, v)]
x_vals = [x]
y_vals = [y(x)]

for i in range(n_steps):
    print(f"{i+1} of {n_steps+1}" )
    s, v, x = rk4_step(s, v, x, dt)
    t = (i+1)*dt
    t_vals.append(t)
    s_vals.append(s)
    v_vals.append(v)
    x_vals.append(x)
    y_vals.append(y(x))
    a_vals.append(acceleration(x, v))

# tonen van de grafieken
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(t_vals, s_vals, label='s(t)', color='blue')
plt.ylabel('Afgelegde afstand (m)')
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(t_vals, v_vals, label='v(t)', color='green')
plt.ylabel('Snelheid (m/s)')
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(t_vals, a_vals, label='a(t)', color='red')
plt.ylabel('Versnelling (N/kg)')
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(x_vals, y_vals, label='Traject y(x)', color='purple')
plt.xlabel('Horizontale afstand x (m)')
plt.ylabel('Hoogte y (m)')
plt.grid()

plt.tight_layout()
plt.show()