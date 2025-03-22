To run this, set the `.env.production` file with the URL of the backend server, then:

```
npm install
npm run build
npm run start
```

If you're not using a base path of `/feed`, check `next.config.js` and set the `BASE_PATH` environment variable or basePath to what you're using.
