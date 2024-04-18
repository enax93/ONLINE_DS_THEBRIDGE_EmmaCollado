-- Inserciones para la tabla Clientes
INSERT INTO Clientes (Apellido, Nombre, Direccion, Ciudad, CodigoPostal, Pais, Telefono, CorreoElectronico) 
VALUES 
    ('Gómez', 'María', 'Calle 123', 'Madrid', '12345', 'España', '123456789', 'gomez.maria@gmail.com'),
    ('Pérez', 'Juan', 'Avenida 456', 'Santo Domingo', '54321', 'República Dominicana', '987654321', 'perez.juan@gmail.com'),
    ('Rodríguez', 'Ana', 'Carrera 789', 'La Haya', '67890', 'Países Bajos', '456789012', 'rodriguez.ana@gmail.com');

-- Inserciones para la tabla Empleados
INSERT INTO Empleados (Apellido, Nombre, FechaNacimiento, FechaContratacion, Direccion, Ciudad, CodigoPostal, Pais, Telefono, CorreoElectronico) 
VALUES 
    ('López', 'Carlos', '1990-01-15', '2020-05-20', 'Calle Pintor Velázquez', 'Málaga', '13579', 'España', '111222333', 'lopez.carlos@gmail.com'),
    ('Martínez', 'Laura', '1985-06-20', '2019-11-10', 'Avenida del Parque', 'Barcelona', '24680', 'España', '444555666', 'martinez.laura@gmail.com'),
    ('González', 'Pedro', '1978-12-05', '2021-03-15', 'Camino 357', 'Ámsterdam', '98765', 'Holanda', '777888999', 'gonzalez.pedro@gmail.com');

-- Inserciones para la tabla Pedidos
INSERT INTO Pedidos (ClienteID, EmpleadoID, FechaPedido, DireccionEnvio, CiudadEnvio, CodigoPostalEnvio, PaisEnvio) 
VALUES 
    (1, 1, '2024-03-19', 'Calle Carton', 'Ciudad Caja', '12345', 'Cuba'),
    (2, 2, '2024-03-20', 'Avenida Vidrio', 'Ciudad Botella', '54321', 'Portugal'),
    (3, 3, '2024-03-21', 'Carrera Plastico', 'Ciudad Bolsa', '67890', 'Italia');


-- Inserciones para la tabla Proveedores
INSERT INTO Proveedores (Nombre, Direccion, Ciudad, CodigoPostal, Pais, Telefono, CorreoElectronico, PaginaWeb)
VALUES 
    ('Telas Romero Ivan', 'Calle mala 1', 'Verona', '12345', 'Italia', '123456789', 'correo1@example.com', 'www.example.com'),
    ('Telas Botones Emmanuel', 'Avenida narnia 2', 'Venecia', '23456', 'Italia', '987654321', 'correo2@example.com', 'www.example.com'),
    ('Telas Hilos Raquel', 'Avenida mucha agua 3', 'Sao Paulo 3', '34567', 'Brasil', '123459876', 'correo3@example.com', 'www.example.com'),
    ('Cinta Naim Hilos', 'Calle principal de mi casa', 'Haarlem 4', '45678', 'Netherlands', '987612345', 'correo4@example.com', 'www.example.com'),
    ('Lazos Hegoi Manuel', 'Esto tambien es una dirección', 'Marsella 5', '56789', 'Francia', '543219876', 'correo5@example.com', 'www.example.com');

-- Inserciones para la tabla Categorias
INSERT INTO Categorias (Nombre, Descripcion)
VALUES 
    ('Telas', 'Telas de diferentes tipos'),
    ('Botones', 'Botones de varios materiales'),
    ('Hilos', 'Hilos para coser'),
    ('Cintas', 'Cintas decorativas'),
    ('Lazos', 'Lazos decorativos');

-- Inserciones para la tabla Colores
INSERT INTO Colores (Descripcion)
VALUES 
    ('Blanco'),
    ('Negro'),
    ('Rojo'),
    ('Amarillo'),
    ('Rosa'),
    ('Morado'),
    ('Naranja'),
    ('Gris'),
    ('Beige'),
    ('Turquesa'),
    ('Celeste'),
    ('Violeta'),
    ('Dorado'),
    ('Plateado'),
    ('Magenta'),
    ('Cian'),
    ('Verde lima');

-- Inserciones para la tabla Piezas
INSERT INTO Piezas (Nombre, ProveedorID, CategoriaID, PrecioUnitario, UnidadesEnStock, ColorID) 
VALUES 
    ('Tela algodón estampada', 1, 1, 15.99, 50, 1),
    ('Botón madera natural', 2, 2, 0.50, 100, 2),
    ('Hilo algodón', 3, 3, 2.25, 30, 3);

-- Inserciones para la tabla DetallesPedido
INSERT INTO DetallesPedido (PedidoID, PiezaID, PrecioUnitario, Cantidad) 
VALUES 
    (1, 1, 10.50, 2),
    (2, 2, 5.75, 3),
    (3, 3, 8.20, 1);