#### *nginx + eff ssl notes*

```
server {

  server_name mo.columbari.us www.mo.columbari.us;

  root /home/ubuntu/public_html;

  location /static {
      root /home/ubuntu/data/;
  }

}
```

# :secret:


```
server {
  if ($host = mo.columbari.us) {
      return 301 https://$host$request_uri;
  }

  server_name mo.columbari.us www.mo.columbari.us;

  listen 80;
  return 404;

}
```
