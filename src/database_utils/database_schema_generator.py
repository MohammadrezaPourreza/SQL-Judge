"""This file contains the class for generating database schemas."""

import logging
import random
import re
import sqlite3
from typing import Dict, List, Optional

from .db_info import get_db_schema
from .execution import execute_sql
from .schema import DatabaseSchema
from .schema import get_primary_keys


class DatabaseSchemaGenerator:
  """Generates database schema with optional examples and descriptions.

  Attributes:
      db_id (str): The database identifier.
      db_path (str): The path to the database file.
      add_examples (bool): Flag to indicate whether to add examples.
      schema_structure (DatabaseSchema): The base schema structure.
      schema_with_examples (DatabaseSchema): The schema including examples.
      schema_with_descriptions (DatabaseSchema): The schema including
        descriptions.
  """

  CACHED_DB_SCHEMA = {}

  def __init__(
      self,
      tentative_schema: Optional[DatabaseSchema] = None,
      schema_with_examples: Optional[DatabaseSchema] = None,
      schema_with_descriptions: Optional[DatabaseSchema] = None,
      db_id: Optional[str] = None,
      db_path: Optional[str] = None,
      add_examples: bool = True,
  ):
    """Initializes the database schema generator.

    Args:
      tentative_schema:
      schema_with_examples:
      schema_with_descriptions:
      db_id:
      db_path:
      add_examples:
    """
    self.db_id = db_id
    self.db_path = db_path
    self.add_examples = add_examples
    if self.db_id not in DatabaseSchemaGenerator.CACHED_DB_SCHEMA:
      DatabaseSchemaGenerator._load_schema_into_cache(
          db_id=db_id, db_path=db_path
      )
    self.schema_structure = tentative_schema or DatabaseSchema()
    self.schema_with_examples = schema_with_examples or DatabaseSchema()
    self.schema_with_descriptions = schema_with_descriptions or DatabaseSchema()
    self._initialize_schema_structure()

  @staticmethod
  def _set_primary_keys(db_path: str, database_schema: DatabaseSchema) -> None:
    """Sets primary keys in the database schema.

    Args:
        db_path (str): The path to the database file.
        database_schema (DatabaseSchema): The database schema to update.
    """
    schema_with_primary_keys = {}
    for table_name in database_schema.tables.keys():
      schema_with_primary_keys[table_name] = {}
      for col in execute_sql(db_path, f"PRAGMA table_info(`{table_name}`)"):
        if col[5] > 0:
          schema_with_primary_keys[table_name][col[1]] = {"primary_key": True}
    database_schema.set_columns_info(schema_with_primary_keys)

  @staticmethod
  def _set_foreign_keys(db_path: str, database_schema: DatabaseSchema) -> None:
    """Sets foreign keys in the database schema.

    Args:
        db_path (str): The path to the database file.
        database_schema (DatabaseSchema): The database schema to update.
    """
    schema_with_references = {}
    for table_name, table_schema in database_schema.tables.items():
      schema_with_references[table_name] = {}
      for column_name in table_schema.columns.keys():
        schema_with_references[table_name][column_name] = {
            "foreign_keys": [],
            "referenced_by": [],
        }

    for table_name, _ in schema_with_references.items():
      query = f"PRAGMA foreign_key_list(`{table_name}`)"
      foreign_keys_info = execute_sql(db_path, query)
      for fk in foreign_keys_info:
        try:
          source_table = table_name
          source_column = database_schema.get_actual_column_name(
              table_name, fk[3]
          )
          destination_table = database_schema.get_actual_table_name(fk[2])
          if not fk[4]:
            destination_column = get_primary_keys(
                database_schema.tables[destination_table]
            )[0]
          else:
            destination_column = database_schema.get_actual_column_name(
                fk[2], fk[4]
            )

          schema_with_references[source_table][source_column][
              "foreign_keys"
          ].append((destination_table, destination_column))
          schema_with_references[destination_table][destination_column][
              "referenced_by"
          ].append((source_table, source_column))
        except Exception as e:
          continue

    database_schema.set_columns_info(schema_with_references)

  @classmethod
  def _load_schema_into_cache(cls, db_id: str, db_path: str) -> None:
    """Loads database schema into cache.

    Args:
        db_id (str): The database identifier.
        db_path (str): The path to the database file.
    """
    db_schema = DatabaseSchema.from_schema_dict(get_db_schema(db_path))
    schema_with_type = {}
    for table_name in db_schema.tables.keys():
      query = f"PRAGMA table_info(`{table_name}`)"
      columns = execute_sql(db_path, query, fetch="all")
      schema_with_type[table_name] = {}
      for col in columns:
        schema_with_type[table_name][col[1]] = {"type": col[2]}
        query = (
            f"SELECT COUNT(*) FROM (SELECT DISTINCT `{col[1]}` FROM"
            f" `{table_name}` LIMIT 21) AS subquery;"
        )
        unique_values = execute_sql(db_path, query, "all", 480)
        is_categorical = int(unique_values[0][0]) < 20
        unique_values = None
        if is_categorical:
          query = (
              f"SELECT DISTINCT `{col[1]}` FROM `{table_name}` WHERE"
              f" `{col[1]}` IS NOT NULL"
          )
          unique_values = execute_sql(db_path, query)
        schema_with_type[table_name][col[1]].update(
            {"unique_values": unique_values}
        )
        try:
          value_statics_query = f"""
          SELECT 'Total count ' || COUNT(`{col[1]}`) || ' - Distinct count ' || COUNT(DISTINCT `{col[1]}`) || 
              ' - Null count ' || SUM(CASE WHEN `{col[1]}` IS NULL THEN 1 ELSE 0 END) AS counts  
          FROM (SELECT `{col[1]}` FROM `{table_name}` LIMIT 100000) AS limited_dataset;
          """
          value_statics = execute_sql(db_path, value_statics_query, "all", 480)
          schema_with_type[table_name][col[1]].update({
              "value_statics": (
                  str(value_statics[0][0]) if value_statics else None
              )
          })
        except sqlite3.Error as e:
          print(f"Error while fetching statistics {table_name}.{col[1]}: {e}")
          schema_with_type[table_name][col[1]].update({"value_statics": None})
          schema_with_type[table_name][col[1]].update({"value_statics": None})
    db_schema.set_columns_info(schema_with_type)
    cls.CACHED_DB_SCHEMA[db_id] = db_schema
    cls._set_primary_keys(db_path, cls.CACHED_DB_SCHEMA[db_id])
    cls._set_foreign_keys(db_path, cls.CACHED_DB_SCHEMA[db_id])

  def _initialize_schema_structure(self) -> None:
    """Initializes the schema structure with table and column info, examples, and descriptions."""
    self._load_table_and_column_info()
    self._load_column_examples()
    self._load_column_descriptions()

  def _load_table_and_column_info(self) -> None:
    """Loads table and column information from cached schema."""
    self.schema_structure = DatabaseSchemaGenerator.CACHED_DB_SCHEMA[
        self.db_id
    ].subselect_schema(self.schema_structure)
    self.schema_structure.add_info_from_schema(
        schema=self.CACHED_DB_SCHEMA[self.db_id],
        field_names=["type", "primary_key", "foreign_keys", "referenced_by"],
    )

  def _load_column_examples(self) -> None:
    """Loads examples for columns in the schema."""
    self.schema_structure.add_info_from_schema(
        schema=self.schema_with_examples, field_names=["examples"]
    )
    for table_name, table_schema in self.schema_structure.tables.items():
      for column_name, column_info in table_schema.columns.items():
        if not column_info.examples:
          examples = (
              DatabaseSchemaGenerator.CACHED_DB_SCHEMA[self.db_id]
              .get_column_info(table_name, column_name)
              .unique_values
          )
          if examples:
            column_info.examples = [str(x[0]) for x in examples][:5]

        if (self.add_examples and not column_info.examples) or (
            (column_info.type.lower()) == "date"
            or ("date" in column_name.lower())
        ):
          query = (
              f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE"
              f" `{column_name}` IS NOT NULL LIMIT 3"
          )
          example = execute_sql(db_path=self.db_path, sql=query, fetch="all")
          if example and len(str(example[0])) < 50:
            column_info.examples = example

        if not column_info.value_statics:
          value_statics = (
              DatabaseSchemaGenerator.CACHED_DB_SCHEMA[self.db_id]
              .get_column_info(table_name, column_name)
              .value_statics
          )
          if value_statics:
            column_info.value_statics = value_statics

  def _load_column_descriptions(self) -> None:
    """Loads descriptions for columns in the schema."""
    self.schema_structure.add_info_from_schema(
        self.schema_with_descriptions,
        field_names=[
            "original_column_name",
            "column_name",
            "column_description",
            "data_format",
            "value_description",
        ],
    )

  def _extract_create_ddl_commands(self) -> Dict[str, str]:
    """Extracts DDL commands to create tables in the schema.

    Returns:
        Dict[str, str]: A dictionary mapping table names to their DDL commands.
    """
    ddl_commands = {}
    for table_name in self.schema_structure.tables.keys():
      query = (
          "SELECT sql FROM sqlite_master WHERE type='table' AND"
          f" name='{table_name}';"
      )
      create_prompt = execute_sql(db_path=self.db_path, sql=query, fetch="one")
      ddl_commands[table_name] = create_prompt[0] if create_prompt else ""
    return ddl_commands

  @staticmethod
  def _separate_column_definitions(column_definitions: str) -> List[str]:
    """Separates column definitions in a DDL command.

    Args:
        column_definitions (str): The column definitions as a single string.

    Returns:
        List[str]: A list of individual column definitions.
    """
    paranthesis_open = 0
    start_position = 0
    definitions = []
    for index, char in enumerate(column_definitions):
      if char == "(":
        paranthesis_open += 1
      elif char == ")":
        paranthesis_open -= 1
      if paranthesis_open == 0 and char == ",":
        definitions.append(column_definitions[start_position:index].strip())
        start_position = index + 1
    definitions.append(column_definitions[start_position:].strip())
    return definitions

  def _is_connection(self, table_name: str, column_name: str) -> bool:
    """Checks if a column is a connection (primary key or foreign key).

    Args:
        table_name (str): The name of the table.
        column_name (str): The name of the column.

    Returns:
        bool: True if the column is a connection, False otherwise.
    """
    column_info = self.CACHED_DB_SCHEMA[self.db_id].get_column_info(
        table_name, column_name
    )
    if column_info is None:
      return False
    if column_info.primary_key:
      return True
    for target_table, _ in column_info.foreign_keys:
      if self.schema_structure.get_table_info(target_table):
        return True
    for target_table, _ in column_info.referenced_by:
      if self.schema_structure.get_table_info(target_table):
        return True
    for target_table_name, table_schema in self.schema_structure.tables.items():
      if table_name.lower() == target_table_name.lower():
        continue
      for (
          target_column_name,
          target_column_info,
      ) in table_schema.columns.items():
        if (
            target_column_name.lower() == column_name.lower()
            and target_column_info.primary_key
        ):
          return True
    return False

  def _get_connections(self) -> Dict[str, List[str]]:
    """Retrieves connections between tables in the schema.

    Returns:
        Dict[str, List[str]]: A dictionary mapping table names to lists of
            connected columns.
    """
    connections = {}
    for table_name, _ in self.schema_structure.tables.items():
      connections[table_name] = []
      for column_name, _ in (
          self.CACHED_DB_SCHEMA[self.db_id].tables[table_name].columns.items()
      ):
        if self._is_connection(table_name, column_name):
          connections[table_name].append(column_name)
    return connections

  def get_schema_with_connections(self) -> Dict[str, List[str]]:
    """Gets schema with connections included.

    Returns:
        Dict[str, List[str]]: The schema with connections included.
    """
    schema_structure_dict = self.schema_structure.to_dict()
    connections = self._get_connections()
    for table_name, connected_columns in connections.items():
      for column_name in connected_columns:
        if column_name.lower() not in [
            col.lower() for col in schema_structure_dict[table_name]
        ]:
          schema_structure_dict[table_name].append(column_name)
    return schema_structure_dict

  def _get_example_column_name_description(
      self,
      table_name: str,
      column_name: str,
      include_value_description: bool = True,
  ) -> str:
    """Retrieves example values and descriptions for a column.

    Args:
        table_name (str): The name of the table.
        column_name (str): The name of the column.
        include_value_description (bool): Flag to include value description.

    Returns:
        str: The example values and descriptions for the column.
    """
    example_part = ""
    name_string = ""
    description_string = ""
    value_statics_string = ""
    value_description_string = ""

    column_info = self.schema_structure.get_column_info(table_name, column_name)
    if column_info:
      if column_info.examples:
        example_part = (
            " Example Values: "
            f"{', '.join([f'`{str(x)}`' for x in column_info.examples])}"
        )
      if column_info.value_statics:
        value_statics_string = f" Value Statics: {column_info.value_statics}"
      if column_info.column_name:
        if (column_info.column_name.lower() != column_name.lower()) and (
            column_info.column_name.strip()
        ):
          name_string = f"| Column Name Meaning: {column_info.column_name}"
      if column_info.column_description:
        description_string = (
            f"| Column Description: {column_info.column_description}"
        )
      if column_info.value_description and include_value_description:
        value_description_string = (
            f"| Value Description: {column_info.value_description}"
        )

    description_part = (
        f"{name_string} {description_string} {value_description_string}"
    )
    joint_string = (
        f" --{example_part} |{value_statics_string} {description_part}"
        if example_part and description_part
        else f" --{example_part or description_part or value_statics_string}"
    )
    if joint_string == " --":
      joint_string = ""
    return joint_string.replace("\n", " ") if joint_string else ""

  def generate_schema_string(
      self,
      include_value_description: bool = True,
      shuffle_cols: bool = True,
      shuffle_tables: bool = True,
  ) -> str:
    """Generates a schema string with descriptions and examples.

    Args:
        include_value_description (bool): Flag to include value descriptions.
        shuffle_cols (bool): Flag to shuffle columns.
        shuffle_tables (bool): Flag to shuffle tables.

    Returns:
        str: The generated schema string.
    """
    ddl_commands = self._extract_create_ddl_commands()
    if shuffle_tables:
      ddl_tables = list(ddl_commands.keys())
      random.shuffle(ddl_tables)
      ddl_commands = {
          table_name: ddl_commands[table_name] for table_name in ddl_tables
      }
    for table_name, ddl_command in ddl_commands.items():
      ddl_command = re.sub(r"\s+", " ", ddl_command.strip())
      create_table_match = re.match(
          r'CREATE TABLE "?`?([\w -]+)`?"?\s*\((.*)\)', ddl_command, re.DOTALL
      )
      if not create_table_match:
        logging.warning("Failed to parse create table command: %s", ddl_command)
        continue
      table = create_table_match.group(1).strip()
      if table != table_name:
        logging.warning("Table name mismatch: %s != %s", table, table_name)
      column_definitions = create_table_match.group(2).strip()
      targeted_columns = self.schema_structure.tables[table_name].columns
      schema_lines = [f"CREATE TABLE {table_name}", "("]
      definitions = DatabaseSchemaGenerator._separate_column_definitions(
          column_definitions
      )
      if shuffle_cols:
        definitions = random.sample(definitions, len(definitions))
      for column_def in definitions:
        column_def = column_def.strip()
        if any(
            keyword in column_def.lower()
            for keyword in ["foreign key", "primary key"]
        ):
          if "primary key" in column_def.lower():
            new_column_def = f"\t{column_def},"
            schema_lines.append(new_column_def)
          if "foreign key" in column_def.lower():
            for t_name in self.schema_structure.tables.keys():
              if t_name.lower() in column_def.lower():
                new_column_def = f"\t{column_def},"
                schema_lines.append(new_column_def)
        else:
          if column_def.startswith("--"):
            continue
          if column_def.startswith("`"):
            column_name = column_def.split("`")[1]
          elif column_def.startswith('"'):
            column_name = column_def.split('"')[1]
          else:
            column_name = column_def.split(" ")[0]

          if (column_name in targeted_columns) or self._is_connection(
              table_name, column_name
          ):
            new_column_def = f"\t{column_def},"
            new_column_def += self._get_example_column_name_description(
                table_name, column_name, include_value_description
            )
            schema_lines.append(new_column_def)
          elif column_def.lower().startswith("unique"):
            new_column_def = f"\t{column_def},"
            schema_lines.append(new_column_def)
      schema_lines.append(");")
      ddl_commands[table_name] = "\n".join(schema_lines)
    return "\n\n".join(ddl_commands.values())

  def get_column_profiles(
      self, with_keys: bool = False, with_references: bool = False
  ) -> Dict[str, Dict[str, str]]:
    """Retrieves profiles for columns in the schema.

    The output is a dictionary with table names as keys mapping to dictionaries
    with column names as keys and column profiles as values.

    Args:
        with_keys (bool): Flag to include primary keys and foreign keys.
        with_references (bool): Flag to include referenced columns.

    Returns:
        Dict[str, Dict[str, str]]: The column profiles.
    """
    column_profiles = {}
    for table_name, table_schema in self.schema_structure.tables.items():
      column_profiles[table_name] = {}
      for column_name, column_info in table_schema.columns.items():
        if with_keys or not (
            column_info.primary_key
            or column_info.foreign_keys
            or column_info.referenced_by
        ):
          column_profile = (
              f"Table name: `{table_name}`\nOriginal column name:"
              f" `{column_name}`\n"
          )
          if (
              column_info.column_name.lower().strip()
              != column_name.lower().strip()
          ) and (column_info.column_name.strip()):
            column_profile += (
                f"Expanded column name: `{column_info.column_name}`\n"
            )
          if column_info.type:
            column_profile += f"Data type: {column_info.type}\n"
          if column_info.column_description:
            column_profile += f"Description: {column_info.column_description}\n"
          if column_info.value_description:
            column_profile += (
                f"Value description: {column_info.value_description}\n"
            )
          if column_info.examples:
            column_profile += (
                "Example of values in the column:"
                f" {', '.join([f'`{str(x)}`' for x in column_info.examples])}\n"
            )
          if column_info.primary_key:
            column_profile += "This column is a primary key.\n"
          if with_references:
            if column_info.foreign_keys:
              column_profile += (
                  "This column references the following columns:\n"
              )
              for target_table, target_column in column_info.foreign_keys:
                column_profile += (
                    f"    Table: `{target_table}`, Column: `{target_column}`\n"
                )
            if column_info.referenced_by:
              column_profile += (
                  "This column is referenced by the following columns:\n"
              )
              for source_table, source_column in column_info.referenced_by:
                column_profile += (
                    f"    Table: `{source_table}`, Column: `{source_column}`\n"
                )
          column_profiles[table_name][column_name] = column_profile
    return column_profiles