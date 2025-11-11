# ------------------ STAGE 1: BUILD ------------------
FROM node:20-alpine AS build
WORKDIR /app

# Copia package.json y package-lock.json para instalar dependencias primero
COPY package*.json ./
RUN npm install

# Copia el resto del código
COPY . .

# Argumento para la URL de la API y variable de entorno
# El backend correrá en http://backend:8000 dentro de la red Docker
ARG VITE_API_URL=http://backend:8000
ENV VITE_API_URL=$VITE_API_URL

# Configuración de Tailwind CSS: asegúrate de que el CLI esté en el PATH
RUN npm install -g @tailwindcss/cli

# Ejecuta la construcción de la aplicación Vue (Vite)
RUN npm run build


# ------------------ STAGE 2: RUN --------------------
FROM nginx:alpine
# Copia la configuración de Nginx
COPY nginx.conf /etc/nginx/conf.d/default.conf
# Copia los archivos estáticos generados por Vite a la carpeta de Nginx
COPY --from=build /app/dist /usr/share/nginx/html

# Nginx expone el puerto 80 por defecto
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]