
**Arrancar la Máquina Virtual**

Una vez importado y si todo ha ido bien, cuando arranquemos la máquina nos aparecerá una pantalla solicitando usuario y password. El usuario y el password de la máquina virtual son: root / student.

Para parar la máquina (una vez habéis hecho login y estáis dentro) podéis utilizar el siguiente comando:

```sh
sd-GCDA:~# poweroff
```

**Cómo acceder a la máquina Virtual mediante ssh y sftp**

Para trabajar con la máquina virtual es más práctico poder conectarnos desde el sistema operativo host (o anfitrión) utilizando ssh. Para poder hacerlo la máquina virtual viene preconfigurada con NAT de forma que el puerto 2522 del host redirige al puerto 22 del guest (máquina virtual).




```sh
➜  ~ ssh -p 2522 root@localhost

The authenticity of host '[localhost]:2522 ([127.0.0.1]:2522)' can't be established.
ED25519 key fingerprint is SHA256:F/ODHPgYMH+pC4ZWu2RAF5VXUhYQ7n6stx25SS+sgQE.
This key is not known by any other names.
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Warning: Permanently added '[localhost]:2522' (ED25519) to the list of known hosts.
root@localhost's password: 
Welcome to Alpine!

The Alpine Wiki contains a large amount of how-to guides and general
information about administrating Alpine systems.
See <http://wiki.alpinelinux.org/>.

You can setup the system with the command: setup-alpine

You may change this message by editing /etc/motd.

sd-GCDA:~# 
```


