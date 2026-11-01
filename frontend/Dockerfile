FROM node:22-alpine AS build
WORKDIR /app
COPY frontend/package.json ./
RUN npm install --no-audit --no-fund
COPY frontend/index.html ./
COPY frontend/vite.config.js ./
COPY frontend/src ./src
RUN npm run build
FROM nginx:1.27-alpine
COPY frontend/default.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
