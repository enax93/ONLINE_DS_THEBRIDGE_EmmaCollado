-- Crear tabla Colores
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'Colores') AND type in (N'U'))
BEGIN
    CREATE TABLE Colores (
        ColorID INT PRIMARY KEY IDENTITY,
        Descripcion VARCHAR(70)
    );
END;

-- Crear tabla Categorias
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'Categorias') AND type in (N'U'))
BEGIN
    CREATE TABLE Categorias (
        CategoriaID INT PRIMARY KEY IDENTITY,
        Nombre VARCHAR(70),
        Descripcion VARCHAR(70)
    );
END;

-- Crear tabla Proveedores
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'Proveedores') AND type in (N'U'))
BEGIN
    CREATE TABLE Proveedores (
        ProveedorID INT PRIMARY KEY IDENTITY,
        Nombre VARCHAR(70),
        Direccion VARCHAR(50),
        Ciudad VARCHAR(50),
        CodigoPostal VARCHAR(10),
        Pais VARCHAR(50),
        Telefono VARCHAR(20),
        CorreoElectronico VARCHAR(50),
        PaginaWeb VARCHAR(70)
    );
END;

-- Crear tabla Piezas
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'Piezas') AND type in (N'U'))
BEGIN
    CREATE TABLE Piezas (
        PiezaID INT PRIMARY KEY IDENTITY,
        Nombre VARCHAR(70),
        ProveedorID INT,
        CategoriaID INT,
        PrecioUnitario REAL,
        UnidadesEnStock INT,
        ColorID INT,
        FOREIGN KEY (ProveedorID) REFERENCES Proveedores(ProveedorID),
        FOREIGN KEY (CategoriaID) REFERENCES Categorias(CategoriaID),
        FOREIGN KEY (ColorID) REFERENCES Colores(ColorID)
    );
END;

-- Crear tabla Empleados
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'Empleados') AND type in (N'U'))
BEGIN
    CREATE TABLE Empleados (
        EmpleadoID INT PRIMARY KEY IDENTITY,
        Apellido VARCHAR(70),
        Nombre VARCHAR(70),
        FechaNacimiento DATE,
        FechaContratacion DATE,
        Direccion VARCHAR(50),
        Ciudad VARCHAR(50),
        CodigoPostal VARCHAR(10),
        Pais VARCHAR(50),
        Telefono VARCHAR(20),
        CorreoElectronico VARCHAR(50)
    );
END;

-- Crear tabla Clientes
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'Clientes') AND type in (N'U'))
BEGIN
    CREATE TABLE Clientes (
        ClienteID INT PRIMARY KEY IDENTITY,
        Apellido VARCHAR(70),
        Nombre VARCHAR(70),
        Direccion VARCHAR(50),
        Ciudad VARCHAR(50),
        CodigoPostal VARCHAR(10),
        Pais VARCHAR(50),
        Telefono VARCHAR(20),
        CorreoElectronico VARCHAR(50)
    );
END;

-- Crear tabla Pedidos
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'Pedidos') AND type in (N'U'))
BEGIN
    CREATE TABLE Pedidos (
        PedidoID INT PRIMARY KEY IDENTITY,
        ClienteID INT,
        EmpleadoID INT,
        FechaPedido DATE,
        DireccionEnvio VARCHAR(50),
        CiudadEnvio VARCHAR(50),
        CodigoPostalEnvio VARCHAR(10),
        PaisEnvio VARCHAR(50),
        FOREIGN KEY (ClienteID) REFERENCES Clientes(ClienteID),
        FOREIGN KEY (EmpleadoID) REFERENCES Empleados(EmpleadoID)
    );
END;

-- Crear tabla DetallesPedido
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'DetallesPedido') AND type in (N'U'))
BEGIN
    CREATE TABLE DetallesPedido (
        DetallePedidoID INT PRIMARY KEY IDENTITY,
        PedidoID INT,
        PiezaID INT,
        PrecioUnitario REAL,
        Cantidad INT,
        FOREIGN KEY (PedidoID) REFERENCES Pedidos(PedidoID),
        FOREIGN KEY (PiezaID) REFERENCES Piezas(PiezaID)
    );
END;
